use super::NativeProtocol;
use super::Tool;
use super::invalid;
use super::string;
use crate::error::ApiError;
use serde_json::Value;
use serde_json::json;

// Stable across inventory reorderings and resumes; never lose MCP namespaces or
// confuse a flat function name with a namespaced function of the same spelling.
fn wire_name(namespace: Option<&str>, name: &str) -> String {
    let identity = json!([namespace, name]).to_string();
    let hash = identity.bytes().fold(0xcbf29ce484222325_u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    });
    let suffix: String = name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .take(40)
        .collect();
    format!("dz_{hash:016x}_{suffix}")
}

fn tools(
    input: &[Value],
    namespace: Option<&str>,
    protocol: NativeProtocol,
    output: &mut Vec<Value>,
    registry: &mut Vec<Tool>,
) -> Result<(), ApiError> {
    for item in input {
        if item["type"] == "namespace" {
            let name = string(item, "name")?;
            if namespace.is_some() {
                return Err(invalid("Nested tool namespaces are unsupported"));
            }
            tools(
                item["tools"]
                    .as_array()
                    .ok_or_else(|| invalid("Namespace tools must be an array"))?,
                Some(&name),
                protocol,
                output,
                registry,
            )?;
            continue;
        }
        let custom = item["type"] == "custom";
        if !custom && item["type"] != "function" {
            return Err(invalid(format!(
                "Native adapter cannot represent tool type {}. Use an MCP function tool instead.",
                item["type"]
            )));
        }
        let name = string(item, "name")?;
        let wire_name = wire_name(namespace, &name);
        if registry.iter().any(|t| t.wire_name == wire_name) {
            return Err(invalid("Duplicate or colliding native tool identity"));
        }
        let schema = if custom {
            json!({"type":"object","properties":{"input":{"type":"string"}},"required":["input"],"additionalProperties":false})
        } else {
            item.get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({"type":"object","properties":{}}))
        };
        if protocol == NativeProtocol::AnthropicMessages
            && ["oneOf", "anyOf", "allOf"]
                .iter()
                .any(|k| schema.get(k).is_some())
        {
            return Err(invalid(
                "Anthropic tools require an object schema without top-level oneOf/anyOf/allOf; fix the tool schema before running",
            ));
        }
        let mut description = item["description"].as_str().unwrap_or(&name).to_owned();
        if let Some(namespace) = namespace {
            description = format!("Namespace: {namespace}.\n{description}");
        }
        if custom && let Some(format) = item.get("format") {
            description.push_str(&format!("\nCustom input format: {format}"));
        }
        output.push(if protocol == NativeProtocol::AnthropicMessages { json!({"name":wire_name,"description":description,"input_schema":schema}) } else { json!({"type":"function","function":{"name":wire_name,"description":description,"parameters":schema}}) });
        registry.push(Tool {
            wire_name,
            name,
            namespace: namespace.map(str::to_owned),
            custom,
        });
    }
    Ok(())
}

fn content(input: &Value, protocol: NativeProtocol) -> Result<Value, ApiError> {
    if let Some(text) = input.as_str() {
        return Ok(json!([{"type":"text","text":text}]));
    }
    let mut output = Vec::new();
    for part in input
        .as_array()
        .ok_or_else(|| invalid("Native message content must be text or an array"))?
    {
        match part["type"].as_str() {
            Some("input_text" | "output_text" | "text") => {
                output.push(json!({"type":"text","text":string(part,"text")?}))
            }
            Some("input_image") => {
                let url = string(part, "image_url")?;
                output.push(if protocol == NativeProtocol::AnthropicMessages {
                    let source = if let Some(data) = url.strip_prefix("data:") {
                        let (mime, data) = data
                            .split_once(";base64,")
                            .ok_or_else(|| invalid("Expected a base64 image data URL"))?;
                        json!({"type":"base64","media_type":mime,"data":data})
                    } else if url.starts_with("https://") {
                        json!({"type":"url","url":url})
                    } else {
                        return Err(invalid("Native image must be a data URL or HTTPS URL"));
                    };
                    json!({"type":"image","source":source})
                } else {
                    json!({"type":"image_url","image_url":{"url":url}})
                });
            }
            _ => {
                return Err(invalid(format!(
                    "Native adapter cannot represent content type {}",
                    part["type"]
                )));
            }
        }
    }
    Ok(json!(output))
}

fn append(messages: &mut Vec<Value>, role: &str, blocks: Value) {
    if let Some(last) = messages.last_mut()
        && last["role"] == role
        && let (Some(current), Some(extra)) = (last["content"].as_array_mut(), blocks.as_array())
    {
        current.extend(extra.iter().cloned());
        return;
    }
    messages.push(json!({"role":role,"content":blocks}));
}

pub(crate) fn native_request(
    input: &Value,
    protocol: NativeProtocol,
) -> Result<(Value, Vec<Tool>), ApiError> {
    if input.get("service_tier").is_some_and(|v| !v.is_null())
        || input.get("access_programs").is_some_and(|v| !v.is_null())
    {
        return Err(invalid(
            "OpenAI service tiers and access programs cannot be translated to a native provider",
        ));
    }
    if input["text"].get("format").is_some_and(|v| !v.is_null()) {
        return Err(invalid(
            "Structured output is not implemented by this native adapter yet",
        ));
    }
    if input["tool_choice"]
        .as_str()
        .is_some_and(|v| v != "auto" && v != "none")
    {
        return Err(invalid("Native adapter supports auto or none tool choice"));
    }
    let mut native_tools = Vec::new();
    let mut registry = Vec::new();
    if let Some(list) = input["tools"].as_array() {
        tools(list, None, protocol, &mut native_tools, &mut registry)?;
    }
    let mut messages = Vec::new();
    let instructions = input["instructions"].as_str().unwrap_or_default();
    let mut system = if instructions.is_empty() {
        Vec::new()
    } else {
        vec![json!({"type":"text","text":instructions})]
    };
    if protocol == NativeProtocol::ChatCompletions && !instructions.is_empty() {
        messages.push(json!({"role":"system","content":instructions}));
    }
    for item in input["input"]
        .as_array()
        .ok_or_else(|| invalid("Native input must be an array"))?
    {
        match item["type"].as_str() {
            Some("message") => {
                let role = string(item, "role")?;
                if !["user", "assistant", "system", "developer"].contains(&role.as_str()) {
                    return Err(invalid("Unknown message role"));
                }
                // Provider-neutral instruction roles remain instructions, never user text.
                let role = if role == "developer" { "system" } else { &role };
                if protocol == NativeProtocol::AnthropicMessages && role == "system" {
                    system.extend(
                        content(&item["content"], protocol)?
                            .as_array()
                            .cloned()
                            .unwrap_or_default(),
                    );
                    continue;
                }
                append(&mut messages, role, content(&item["content"], protocol)?);
            }
            Some("function_call" | "custom_tool_call") => {
                let name = string(item, "name")?;
                let namespace = item["namespace"].as_str();
                let custom = item["type"] == "custom_tool_call";
                let arguments = if custom {
                    json!({"input":string(item,"input")?})
                } else {
                    serde_json::from_str::<Value>(&string(item, "arguments")?)
                        .map_err(|_| invalid("Recorded tool arguments are invalid JSON"))?
                };
                let id = string(item, "call_id")?;
                let name = wire_name(namespace, &name);
                if protocol == NativeProtocol::AnthropicMessages {
                    append(
                        &mut messages,
                        "assistant",
                        json!([{"type":"tool_use","id":id,"name":name,"input":arguments}]),
                    );
                } else {
                    messages.push(json!({"role":"assistant","content":null,"tool_calls":[{"id":id,"type":"function","function":{"name":name,"arguments":arguments.to_string()}}]}));
                }
            }
            Some("function_call_output" | "custom_tool_call_output") => {
                let id = string(item, "call_id")?;
                let output = item
                    .get("output")
                    .ok_or_else(|| invalid("Missing tool output"))?;
                let output = content(output, protocol)?;
                if protocol == NativeProtocol::AnthropicMessages {
                    append(
                        &mut messages,
                        "user",
                        json!([{"type":"tool_result","tool_use_id":id,"content":output}]),
                    );
                } else {
                    messages.push(json!({"role":"tool","tool_call_id":id,"content":output}));
                }
            }
            Some("reasoning") => {
                let opaque = item["encrypted_content"].as_str().ok_or_else(|| {
                    invalid(
                        "Cannot replay reasoning from a different provider. Start a fresh thread.",
                    )
                })?;
                let prefix = if protocol == NativeProtocol::AnthropicMessages {
                    "anthropic-native-v1:"
                } else {
                    "chat-native-v1:"
                };
                let value: Value = serde_json::from_str(opaque.strip_prefix(prefix).ok_or_else(|| invalid("Cannot replay reasoning from a different provider. Start a fresh thread."))?).map_err(|_| invalid("Invalid native reasoning state"))?;
                if protocol == NativeProtocol::AnthropicMessages {
                    append(&mut messages, "assistant", json!([value]));
                } else {
                    messages
                        .push(json!({"role":"assistant","content":null,"reasoning_content":value}));
                }
            }
            _ => {
                return Err(invalid(format!(
                    "Native adapter cannot represent history item {}",
                    item["type"]
                )));
            }
        }
    }
    if protocol == NativeProtocol::ChatCompletions {
        let mut joined: Vec<Value> = Vec::new();
        for message in messages {
            if message["role"] == "assistant"
                && let Some(last) = joined.last_mut().filter(|m| m["role"] == "assistant")
            {
                if let Some(extra) = message["content"].as_array() {
                    let mut blocks = last["content"].as_array().cloned().unwrap_or_default();
                    blocks.extend(extra.iter().cloned());
                    last["content"] = json!(blocks);
                }
                if let Some(extra) = message["tool_calls"].as_array() {
                    let mut calls = last["tool_calls"].as_array().cloned().unwrap_or_default();
                    calls.extend(extra.iter().cloned());
                    last["tool_calls"] = json!(calls);
                }
                if let Some(extra) = message["reasoning_content"].as_str() {
                    last["reasoning_content"] = json!(format!(
                        "{}{extra}",
                        last["reasoning_content"].as_str().unwrap_or_default()
                    ));
                }
            } else {
                joined.push(message);
            }
        }
        messages = joined;
    }
    let mut body = json!({"model":string(input,"model")?,"messages":messages,"stream":true});
    if protocol == NativeProtocol::AnthropicMessages {
        body["system"] = json!(system);
        // ponytail: experimental fixed output ceiling; model-specific budgets are the next capability extension.
        body["max_tokens"] = json!(8192);
        if input["reasoning"]["effort"]
            .as_str()
            .is_some_and(|e| e != "none")
        {
            body["thinking"] = json!({"type":"enabled","budget_tokens":4096});
        }
    } else {
        body["stream_options"] = json!({"include_usage":true});
    }
    if input["tool_choice"] != "none" && !native_tools.is_empty() {
        body["tools"] = json!(native_tools);
        body["tool_choice"] = if protocol == NativeProtocol::AnthropicMessages {
            json!({"type":"auto","disable_parallel_tool_use":input["parallel_tool_calls"] == false})
        } else {
            json!("auto")
        };
    }
    Ok((body, registry))
}
