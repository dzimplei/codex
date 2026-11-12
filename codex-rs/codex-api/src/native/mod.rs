//! Opt-in native inference adapters. Codex retains ownership of tools, approvals,
//! history, cancellation and compaction; this module only translates the wire.
use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::error::ApiError;
use codex_client::StreamResponse;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde_json::Value;
use serde_json::json;
use std::collections::BTreeMap;
use std::time::Duration;
use tokio::sync::mpsc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NativeProtocol {
    #[default]
    Responses,
    AnthropicMessages,
    ChatCompletions,
}

#[derive(Clone, Debug)]
pub(crate) struct Tool {
    wire_name: String,
    name: String,
    namespace: Option<String>,
    custom: bool,
}

fn invalid(message: impl Into<String>) -> ApiError {
    ApiError::InvalidRequest {
        message: message.into(),
    }
}

fn string(value: &Value, key: &str) -> Result<String, ApiError> {
    value[key]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| invalid(format!("Native adapter requires {key}")))
}

mod request;
pub(crate) use request::native_request;

#[derive(Default)]
struct Block {
    kind: String,
    id: String,
    name: String,
    text: String,
    arguments: String,
    signature: String,
}
struct Decoder {
    protocol: NativeProtocol,
    tools: Vec<Tool>,
    id: String,
    blocks: BTreeMap<u64, Block>,
    usage: TokenUsage,
    stop: Option<String>,
    complete: bool,
    usage_seen: bool,
}
impl Decoder {
    fn new(protocol: NativeProtocol, tools: Vec<Tool>) -> Self {
        Self {
            protocol,
            tools,
            id: String::new(),
            blocks: BTreeMap::new(),
            usage: TokenUsage::default(),
            stop: None,
            complete: false,
            usage_seen: false,
        }
    }
    fn item(&self, index: u64, block: &Block) -> Result<ResponseItem, ApiError> {
        let id = format!("{}_{index}", self.id);
        let item = match block.kind.as_str() {
            "text" => {
                json!({"type":"message","id":id,"role":"assistant","content":[{"type":"output_text","text":block.text}]})
            }
            "thinking" | "redacted_thinking" => {
                let (prefix, state) = if self.protocol == NativeProtocol::AnthropicMessages {
                    let state = if block.kind == "redacted_thinking" {
                        json!({"type":"redacted_thinking","data":block.text})
                    } else {
                        json!({"type":"thinking","thinking":block.text,"signature":block.signature})
                    };
                    ("anthropic-native-v1:", state)
                } else {
                    ("chat-native-v1:", json!(block.text))
                };
                json!({"type":"reasoning","id":id,"summary":[],"encrypted_content":format!("{prefix}{state}")})
            }
            "tool_use" => {
                let tool = self
                    .tools
                    .iter()
                    .find(|t| t.wire_name == block.name)
                    .ok_or_else(|| invalid("Provider returned a tool that was not offered"))?;
                if block.id.is_empty() {
                    return Err(invalid("Provider tool call has no ID"));
                }
                let args: Value = serde_json::from_str(&block.arguments)
                    .map_err(|_| invalid("Provider tool arguments are incomplete or malformed"))?;
                if !args.is_object() {
                    return Err(invalid("Provider tool arguments must be an object"));
                }
                if tool.custom {
                    json!({"type":"custom_tool_call","id":id,"call_id":block.id,"name":tool.name,"namespace":tool.namespace,"input":string(&args,"input")?})
                } else {
                    json!({"type":"function_call","id":id,"call_id":block.id,"name":tool.name,"namespace":tool.namespace,"arguments":block.arguments})
                }
            }
            _ => return Err(invalid("Unsupported native output block")),
        };
        serde_json::from_value(item)
            .map_err(|e| invalid(format!("Invalid normalized native item: {e}")))
    }
    fn finish(&mut self) -> Result<Vec<ResponseEvent>, ApiError> {
        let stop = self
            .stop
            .as_deref()
            .ok_or_else(|| invalid("Native stream ended without a finish reason"))?;
        let has_tools = self.blocks.values().any(|b| b.kind == "tool_use");
        if self.id.is_empty()
            || !self
                .blocks
                .values()
                .any(|b| b.kind == "tool_use" || (b.kind == "text" && !b.text.is_empty()))
        {
            return Err(invalid(
                "Native provider finished without a response ID, answer or tool call",
            ));
        }
        if ![
            "end_turn",
            "stop_sequence",
            "tool_use",
            "stop",
            "tool_calls",
            "refusal",
        ]
        .contains(&stop)
        {
            return Err(invalid(format!(
                "Native output was not complete ({stop}); partial tool calls will not execute"
            )));
        }
        if has_tools && !["tool_use", "tool_calls"].contains(&stop) {
            return Err(invalid(
                "Native tools were returned with a refusal or non-tool finish reason",
            ));
        }
        // Validate the entire call set before emitting ANY executable item.
        let mut ids = std::collections::HashSet::new();
        for block in self.blocks.values().filter(|b| b.kind == "tool_use") {
            if !ids.insert(&block.id) {
                return Err(invalid("Duplicate native tool-call ID"));
            }
        }
        let mut events = self
            .blocks
            .iter()
            .map(|(i, b)| self.item(*i, b).map(ResponseEvent::OutputItemDone))
            .collect::<Result<Vec<_>, _>>()?;
        self.usage.total_tokens = self.usage.input_tokens + self.usage.output_tokens;
        events.push(ResponseEvent::Completed {
            response_id: self.id.clone(),
            token_usage: self.usage_seen.then(|| self.usage.clone()),
            usage_metadata: None,
            end_turn: Some(!has_tools),
        });
        self.complete = true;
        Ok(events)
    }
    fn event(&mut self, data: &str) -> Result<Vec<ResponseEvent>, ApiError> {
        if self.complete {
            return Ok(Vec::new());
        }
        if data == "[DONE]" {
            return self.finish();
        }
        let v: Value =
            serde_json::from_str(data).map_err(|_| invalid("Malformed native SSE JSON"))?;
        if v.get("error").is_some() || v["type"] == "error" {
            return Err(invalid(format!(
                "Native provider stream error: {}",
                v["error"]["type"].as_str().unwrap_or("upstream_error")
            )));
        }
        let mut out = Vec::new();
        if self.protocol == NativeProtocol::AnthropicMessages {
            let index = v["index"].as_u64().unwrap_or(0);
            match v["type"].as_str() {
                Some("message_start") => {
                    self.id = string(&v["message"], "id")?;
                    self.usage_seen = v["message"]["usage"].is_object();
                    self.usage.input_tokens =
                        v["message"]["usage"]["input_tokens"].as_i64().unwrap_or(0);
                    self.usage.cached_input_tokens =
                        v["message"]["usage"]["cache_read_input_tokens"]
                            .as_i64()
                            .unwrap_or(0);
                    self.usage.cache_write_input_tokens =
                        v["message"]["usage"]["cache_creation_input_tokens"]
                            .as_i64()
                            .unwrap_or(0);
                    self.usage.input_tokens +=
                        self.usage.cached_input_tokens + self.usage.cache_write_input_tokens;
                    self.usage.output_tokens =
                        v["message"]["usage"]["output_tokens"].as_i64().unwrap_or(0);
                    out.push(ResponseEvent::Created {
                        response_id: Some(self.id.clone()),
                    });
                }
                Some("content_block_start") => {
                    let b = &v["content_block"];
                    let kind = string(b, "type")?;
                    if !["text", "tool_use", "thinking", "redacted_thinking"]
                        .contains(&kind.as_str())
                    {
                        return Err(invalid(format!(
                            "Unsupported Anthropic content block {kind}"
                        )));
                    }
                    let text = b["text"]
                        .as_str()
                        .or(b["thinking"].as_str())
                        .or(b["data"].as_str())
                        .unwrap_or_default()
                        .to_owned();
                    let arguments = b["input"]
                        .as_object()
                        .filter(|o| !o.is_empty())
                        .map(|_| b["input"].to_string())
                        .unwrap_or_default();
                    let block = Block {
                        kind,
                        arguments,
                        id: b["id"].as_str().unwrap_or_default().into(),
                        name: b["name"].as_str().unwrap_or_default().into(),
                        text,
                        signature: b["signature"].as_str().unwrap_or_default().into(),
                    };
                    if block.kind == "text" {
                        out.push(ResponseEvent::OutputItemAdded(self.item(
                            index,
                            &Block {
                                kind: "text".into(),
                                ..Default::default()
                            },
                        )?));
                    }
                    if self.blocks.len() >= 1024 || self.blocks.contains_key(&index) {
                        return Err(invalid("Too many or duplicate native content blocks"));
                    }
                    self.blocks.insert(index, block);
                }
                Some("content_block_delta") => {
                    let b = self
                        .blocks
                        .get_mut(&index)
                        .ok_or_else(|| invalid("Native delta has no content block"))?;
                    match v["delta"]["type"].as_str() {
                        Some("text_delta") => {
                            let delta = string(&v["delta"], "text")?;
                            b.text.push_str(&delta);
                            out.push(ResponseEvent::OutputTextDelta(delta));
                        }
                        Some("input_json_delta") => {
                            b.arguments.push_str(&string(&v["delta"], "partial_json")?)
                        }
                        Some("thinking_delta") => {
                            let delta = string(&v["delta"], "thinking")?;
                            b.text.push_str(&delta);
                            out.push(ResponseEvent::ReasoningContentDelta {
                                delta,
                                content_index: index as i64,
                            });
                        }
                        Some("signature_delta") => {
                            b.signature.push_str(&string(&v["delta"], "signature")?)
                        }
                        _ => return Err(invalid("Unsupported Anthropic content delta")),
                    }
                }
                Some("content_block_stop") => {
                    if let Some(b) = self.blocks.get_mut(&index)
                        && b.kind == "tool_use"
                        && b.arguments.is_empty()
                    {
                        b.arguments = "{}".into();
                    }
                }
                Some("message_delta") => {
                    if let Some(reason) = v["delta"]["stop_reason"].as_str() {
                        self.stop = Some(reason.into())
                    }
                    if let Some(tokens) = v["usage"]["output_tokens"].as_i64() {
                        self.usage.output_tokens = tokens
                    }
                }
                Some("message_stop") => return self.finish(),
                Some("ping") => {}
                _ => {} // New non-content metadata events are forward compatible.
            }
        } else {
            if self.id.is_empty()
                && let Some(id) = v["id"].as_str()
            {
                self.id = id.into();
                out.push(ResponseEvent::Created {
                    response_id: Some(id.into()),
                });
            }
            if let Some(usage) = v.get("usage").filter(|u| !u.is_null()) {
                self.usage_seen = true;
                self.usage.input_tokens = usage["prompt_tokens"].as_i64().unwrap_or(0);
                self.usage.output_tokens = usage["completion_tokens"].as_i64().unwrap_or(0);
                self.usage.cached_input_tokens = usage["prompt_tokens_details"]["cached_tokens"]
                    .as_i64()
                    .unwrap_or(0);
            }
            for choice in v["choices"].as_array().into_iter().flatten() {
                if choice["index"].as_u64().unwrap_or(0) != 0 {
                    return Err(invalid(
                        "Native adapter accepts exactly one completion choice",
                    ));
                }
                let delta = &choice["delta"];
                if let Some(text) = delta["content"].as_str() {
                    let fresh = !self.blocks.contains_key(&0);
                    if fresh {
                        out.push(ResponseEvent::OutputItemAdded(self.item(
                            0,
                            &Block {
                                kind: "text".into(),
                                ..Default::default()
                            },
                        )?));
                    }
                    self.blocks
                        .entry(0)
                        .or_insert_with(|| Block {
                            kind: "text".into(),
                            ..Default::default()
                        })
                        .text
                        .push_str(text);
                    out.push(ResponseEvent::OutputTextDelta(text.into()));
                }
                if let Some(text) = delta["reasoning_content"].as_str() {
                    self.blocks
                        .entry(1)
                        .or_insert_with(|| Block {
                            kind: "thinking".into(),
                            ..Default::default()
                        })
                        .text
                        .push_str(text);
                    out.push(ResponseEvent::ReasoningContentDelta {
                        delta: text.into(),
                        content_index: 0,
                    });
                }
                if let Some(text) = delta["refusal"].as_str() {
                    self.blocks
                        .entry(0)
                        .or_insert_with(|| Block {
                            kind: "text".into(),
                            ..Default::default()
                        })
                        .text
                        .push_str(text);
                    out.push(ResponseEvent::OutputTextDelta(text.into()));
                }
                for tool in delta["tool_calls"].as_array().into_iter().flatten() {
                    let index = tool["index"]
                        .as_u64()
                        .filter(|i| *i < 1024)
                        .ok_or_else(|| invalid("Missing or excessive native tool index"))?
                        + 2;
                    let block = self.blocks.entry(index).or_insert_with(|| Block {
                        kind: "tool_use".into(),
                        ..Default::default()
                    });
                    if let Some(id) = tool["id"].as_str() {
                        block.id.push_str(id)
                    }
                    if let Some(name) = tool["function"]["name"].as_str() {
                        block.name.push_str(name)
                    }
                    if let Some(args) = tool["function"]["arguments"].as_str() {
                        block.arguments.push_str(args)
                    }
                }
                if let Some(reason) = choice["finish_reason"].as_str() {
                    self.stop = Some(reason.into())
                }
            }
        }
        Ok(out)
    }
}

pub(crate) fn spawn_native_stream(
    response: StreamResponse,
    protocol: NativeProtocol,
    tools: Vec<Tool>,
    idle: Duration,
) -> ResponseStream {
    let upstream_request_id = response
        .headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let (tx, rx_event) = mpsc::channel(32);
    tokio::spawn(async move {
        let bytes = response.bytes.scan(0usize, |total, chunk| {
            let result = chunk.and_then(|bytes| {
                *total = total.saturating_add(bytes.len());
                if *total > 32 * 1024 * 1024 {
                    Err(codex_client::TransportError::ResponseTooLarge {
                        max_bytes: 32 * 1024 * 1024,
                    })
                } else {
                    Ok(bytes)
                }
            });
            futures::future::ready(Some(result))
        });
        let mut stream = bytes.boxed().eventsource();
        let mut decoder = Decoder::new(protocol, tools);
        loop {
            let next = tokio::select! { _=tx.closed()=>return, next=tokio::time::timeout(idle,stream.next())=>next };
            let events = match next {
                Ok(Some(Ok(event))) => decoder.event(&event.data),
                Ok(Some(Err(_))) => Err(invalid("Native SSE connection failed before completion")),
                Ok(None) if decoder.complete => return,
                Ok(None)
                    if protocol == NativeProtocol::ChatCompletions && decoder.stop.is_some() =>
                {
                    decoder.finish()
                }
                Ok(None) => Err(invalid(
                    "Native SSE closed before its terminal event; no partial tools were executed",
                )),
                Err(_) => Err(ApiError::Transport(codex_client::TransportError::Timeout)),
            };
            match events {
                Ok(events) => {
                    for event in events {
                        if tx.send(Ok(event)).await.is_err() {
                            return;
                        }
                    }
                }
                Err(error) => {
                    let _ = tx.send(Err(error)).await;
                    return;
                }
            }
            if decoder.complete {
                return;
            }
        }
    });
    ResponseStream {
        rx_event,
        upstream_request_id,
        interrupt: None,
    }
}

#[cfg(test)]
#[path = "native_tests.rs"]
mod tests;
