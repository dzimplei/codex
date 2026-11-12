use super::*;
use pretty_assertions::assert_eq;

fn request() -> Value {
    json!({"model":"fixture-model","instructions":"System instructions","input":[{"type":"message","role":"developer","content":[{"type":"input_text","text":"Workspace instructions"}]},{"type":"message","role":"user","content":[{"type":"input_text","text":"Inspect the page"}]}],"tools":[{"type":"namespace","name":"browser","tools":[{"type":"function","name":"read","parameters":{"type":"object","properties":{}}}]},{"type":"custom","name":"apply_patch"}],"tool_choice":"auto","parallel_tool_calls":true})
}
fn anthropic_start(d: &mut Decoder) {
    d.event(&json!({"type":"message_start","message":{"id":"msg-fixture","usage":{"input_tokens":10,"cache_read_input_tokens":30,"cache_creation_input_tokens":20,"output_tokens":1}}}).to_string()).unwrap();
}
fn anthropic_tool(d: &mut Decoder, index: u64, name: &str, args: &str) {
    d.event(&json!({"type":"content_block_start","index":index,"content_block":{"type":"tool_use","id":format!("call-{index}"),"name":name,"input":{}}}).to_string()).unwrap();
    d.event(&json!({"type":"content_block_delta","index":index,"delta":{"type":"input_json_delta","partial_json":args}}).to_string()).unwrap();
}
#[test]
fn native_requests_preserve_instructions_namespace_and_custom_tools() {
    let (body, tools) = native_request(&request(), NativeProtocol::AnthropicMessages).unwrap();
    assert_eq!(
        body["system"],
        json!([{"type":"text","text":"System instructions"},{"type":"text","text":"Workspace instructions"}])
    );
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(tools[0].namespace.as_deref(), Some("browser"));
    let mut collision = request();
    collision["tools"]
        .as_array_mut()
        .unwrap()
        .push(json!({"type":"function","name":"browser__read","parameters":{"type":"object"}}));
    let (_, names) = native_request(&collision, NativeProtocol::AnthropicMessages).unwrap();
    assert_ne!(names[0].wire_name, names[2].wire_name);
    assert_eq!(
        body["tools"][1]["input_schema"]["properties"]["input"]["type"],
        "string"
    );
    assert_eq!(body["max_tokens"], 8192);
    assert!(body.get("store").is_none());
}
#[test]
fn unknown_builtin_and_unsupported_content_fail_before_http() {
    let mut r = request();
    r["tools"] = json!([{"type":"web_search"}]);
    assert!(matches!(
        native_request(&r, NativeProtocol::AnthropicMessages),
        Err(ApiError::InvalidRequest { .. })
    ));
    r = request();
    r["service_tier"] = json!("priority");
    assert!(native_request(&r, NativeProtocol::ChatCompletions).is_err());
    r = request();
    r["input"][1]["content"] = json!([{"type":"input_audio","data":"opaque"}]);
    assert!(native_request(&r, NativeProtocol::ChatCompletions).is_err());
    r = request();
    r["tools"][0]["tools"][0]["parameters"] = json!({"anyOf":[{},{}]});
    assert!(native_request(&r, NativeProtocol::AnthropicMessages).is_err());
}
#[test]
fn tools_are_validated_atomically_and_preserve_namespace_and_arguments() {
    let (_, tools) = native_request(&request(), NativeProtocol::AnthropicMessages).unwrap();
    let name = tools[0].wire_name.clone();
    let custom = tools[1].wire_name.clone();
    let mut d = Decoder::new(NativeProtocol::AnthropicMessages, tools);
    anthropic_start(&mut d);
    anthropic_tool(&mut d, 0, &name, "{\"tabId\":\"tab-1\"}");
    anthropic_tool(
        &mut d,
        1,
        &custom,
        "{\"input\":\"*** Begin Patch\\n*** End Patch\"}",
    );
    d.event(&json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}).to_string()).unwrap();
    let events = d.event("{\"type\":\"message_stop\"}").unwrap();
    let ResponseEvent::OutputItemDone(item) = &events[0] else {
        panic!("tool item")
    };
    let item = serde_json::to_value(item).unwrap();
    assert_eq!(item["namespace"], "browser");
    assert_eq!(item["name"], "read");
    assert_eq!(item["arguments"], "{\"tabId\":\"tab-1\"}");
    let ResponseEvent::OutputItemDone(item) = &events[1] else {
        panic!("custom item")
    };
    assert_eq!(
        serde_json::to_value(item).unwrap()["type"],
        "custom_tool_call"
    );
    let ResponseEvent::Completed {
        token_usage: Some(usage),
        end_turn,
        ..
    } = events.last().unwrap()
    else {
        panic!("usage")
    };
    assert_eq!(
        (
            usage.input_tokens,
            usage.cached_input_tokens,
            usage.cache_write_input_tokens,
            usage.total_tokens,
            *end_turn
        ),
        (60, 30, 20, 67, Some(false))
    );
}
#[test]
fn partial_refused_or_unoffered_tools_never_become_executable_items() {
    for (name, args, stop) in [
        ("offered", "{}", "max_tokens"),
        ("offered", "{", "tool_use"),
        ("not-offered", "{}", "tool_use"),
        ("offered", "{}", "refusal"),
    ] {
        let (_, tools) = native_request(&request(), NativeProtocol::AnthropicMessages).unwrap();
        let offered = tools[0].wire_name.clone();
        let mut d = Decoder::new(NativeProtocol::AnthropicMessages, tools);
        anthropic_start(&mut d);
        anthropic_tool(
            &mut d,
            0,
            if name == "offered" { &offered } else { name },
            args,
        );
        d.stop = Some(stop.into());
        assert!(matches!(d.finish(), Err(ApiError::InvalidRequest { .. })));
    }
}
#[test]
fn parallel_chat_tools_and_reasoning_replay_share_one_assistant_message() {
    let mut r = request();
    r["input"] = json!([
        {"type":"reasoning","summary":[],"encrypted_content":"chat-native-v1:\"analysis\""},
        {"type":"function_call","namespace":"browser","name":"read","call_id":"a","arguments":"{}"},
        {"type":"function_call","namespace":"browser","name":"read","call_id":"b","arguments":"{}"},
        {"type":"function_call_output","call_id":"a","output":"one"},
        {"type":"function_call_output","call_id":"b","output":"two"}
    ]);
    let (body, _) = native_request(&r, NativeProtocol::ChatCompletions).unwrap();
    assert_eq!(body["messages"][1]["reasoning_content"], "analysis");
    assert_eq!(
        body["messages"][1]["tool_calls"].as_array().unwrap().len(),
        2
    );
    assert_eq!(body["messages"][2]["role"], "tool");
    assert!(native_request(&r, NativeProtocol::AnthropicMessages).is_err());
}
#[test]
fn signed_thinking_is_replayed_verbatim() {
    let (_, tools) = native_request(&request(), NativeProtocol::AnthropicMessages).unwrap();
    let mut d = Decoder::new(NativeProtocol::AnthropicMessages, tools);
    anthropic_start(&mut d);
    d.event(&json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"Keep me","signature":"original-signature"}}).to_string()).unwrap();
    d.blocks.insert(
        1,
        Block {
            kind: "text".into(),
            text: "Done".into(),
            ..Default::default()
        },
    );
    d.stop = Some("end_turn".into());
    let events = d.finish().unwrap();
    let ResponseEvent::OutputItemDone(item) = &events[0] else {
        panic!("reasoning")
    };
    let mut r = request();
    r["input"] = json!([serde_json::to_value(item).unwrap()]);
    let (body, _) = native_request(&r, NativeProtocol::AnthropicMessages).unwrap();
    assert_eq!(
        body["messages"][0]["content"][0],
        json!({"type":"thinking","thinking":"Keep me","signature":"original-signature"})
    );
}
#[test]
fn chat_usage_can_follow_finish_and_tool_arguments_can_span_events() {
    let (_, tools) = native_request(&request(), NativeProtocol::ChatCompletions).unwrap();
    let name = tools[0].wire_name.clone();
    let mut d = Decoder::new(NativeProtocol::ChatCompletions, tools);
    for value in [
        json!({"id":"chat-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":name,"arguments":"{\"x\":"}}]}}]}),
        json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":"tool_calls"}]}),
        json!({"choices":[],"usage":{"prompt_tokens":50,"completion_tokens":8,"prompt_tokens_details":{"cached_tokens":20}}}),
    ] {
        assert!(
            d.event(&value.to_string())
                .unwrap()
                .iter()
                .all(|e| !matches!(e, ResponseEvent::OutputItemDone(_)))
        )
    }
    let events = d.event("[DONE]").unwrap();
    let ResponseEvent::Completed {
        token_usage: Some(usage),
        ..
    } = events.last().unwrap()
    else {
        panic!("usage")
    };
    assert_eq!(usage.total_tokens, 58);
}

#[tokio::test]
async fn fragmented_sse_uses_existing_http_client_and_emits_native_text() {
    use crate::auth::AuthProvider;
    use crate::endpoint::responses::ResponsesClient;
    use crate::requests::Compression;
    use codex_client::Provider;
    use codex_client::ReqwestTransport;
    use codex_client::RetryConfig;
    use std::sync::Arc;
    use wiremock::Mock;
    use wiremock::MockServer;
    use wiremock::ResponseTemplate;
    use wiremock::matchers::body_partial_json;
    use wiremock::matchers::header;
    use wiremock::matchers::method;
    use wiremock::matchers::path;
    struct NoAuth;
    impl AuthProvider for NoAuth {
        fn add_auth_headers(&self, _: &mut http::HeaderMap) {}
    }
    for (protocol, route, body) in [
        (
            NativeProtocol::AnthropicMessages,
            "/v1/messages",
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"a\",\"usage\":{\"input_tokens\":5}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ),
        (
            NativeProtocol::ChatCompletions,
            "/v1/chat/completions",
            "data: {\"id\":\"a\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
        ),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(route))
            .and(header("accept", "text/event-stream"))
            .and(body_partial_json(
                json!({"model":"fixture-model","stream":true}),
            ))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .expect(1)
            .mount(&server)
            .await;
        let provider = Provider {
            name: "fixture".into(),
            base_url: format!("{}/v1", server.uri()),
            query_params: None,
            headers: Default::default(),
            retry: RetryConfig {
                max_attempts: 1,
                base_delay: Duration::from_millis(1),
                retry_429: false,
                retry_5xx: false,
                retry_transport: false,
            },
            stream_idle_timeout: Duration::from_secs(3),
        };
        let client = ResponsesClient::new(
            ReqwestTransport::from_http_client(
                codex_http_client::HttpClientBuilder::new()
                    .build_direct()
                    .unwrap(),
            ),
            provider,
            Arc::new(NoAuth),
        )
        .with_protocol(protocol);
        let mut stream = client
            .stream(request(), Default::default(), Compression::None, None)
            .await
            .unwrap();
        let mut text = String::new();
        let mut completed = false;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                ResponseEvent::OutputTextDelta(delta) => text.push_str(&delta),
                ResponseEvent::Completed { .. } => completed = true,
                _ => {}
            }
        }
        assert_eq!((text, completed), ("Hello".into(), true));
    }
}

#[tokio::test]
async fn truncated_native_sse_is_an_error_without_executable_tools() {
    use bytes::Bytes;
    let bytes = futures::stream::iter([Ok(Bytes::from_static(
        b"data: {\"id\":\"a\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n",
    ))])
    .boxed();
    let response = StreamResponse {
        status: http::StatusCode::OK,
        headers: Default::default(),
        bytes,
    };
    let mut stream = spawn_native_stream(
        response,
        NativeProtocol::ChatCompletions,
        Vec::new(),
        Duration::from_secs(1),
    );
    let mut failed = false;
    while let Some(event) = stream.next().await {
        match event {
            Err(ApiError::InvalidRequest { .. }) => failed = true,
            Ok(ResponseEvent::OutputItemDone(_) | ResponseEvent::Completed { .. }) => {
                panic!("truncated stream completed")
            }
            _ => {}
        }
    }
    assert!(failed);
}

#[tokio::test]
async fn dropping_a_native_turn_cancels_its_pending_network_stream() {
    use bytes::Bytes;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;
    struct Dropped(Arc<AtomicBool>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let dropped = Arc::new(AtomicBool::new(false));
    let guard = Dropped(dropped.clone());
    let bytes = futures::stream::pending::<Result<Bytes, codex_client::TransportError>>()
        .map(move |chunk| {
            let _ = &guard;
            chunk
        })
        .boxed();
    let response = StreamResponse {
        status: http::StatusCode::OK,
        headers: Default::default(),
        bytes,
    };
    let stream = spawn_native_stream(
        response,
        NativeProtocol::ChatCompletions,
        Vec::new(),
        Duration::from_secs(60),
    );
    drop(stream);
    tokio::time::timeout(Duration::from_secs(1), async {
        while !dropped.load(Ordering::SeqCst) {
            tokio::task::yield_now().await
        }
    })
    .await
    .expect("cancelled network stream was dropped");
}
