use std::collections::HashMap;

use crate::proto;

fn regex_escape_literal(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        if matches!(
            character,
            '.' | '+' | '*' | '?' | '^' | '$' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\'
        ) {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

/// Convert proto SamplingParams to a serde_json map (used as Python dict via PyO3).
#[allow(deprecated)]
fn sampling_params_to_map(
    params: &Option<proto::SamplingParams>,
) -> Result<serde_json::Value, String> {
    match params {
        Some(p) => {
            let mut map = serde_json::Map::new();
            if let Some(v) = p.temperature {
                map.insert("temperature".into(), serde_json::json!(v));
            }
            if let Some(v) = p.top_p {
                map.insert("top_p".into(), serde_json::json!(v));
            }
            if let Some(v) = p.top_k {
                map.insert("top_k".into(), serde_json::json!(v));
            }
            if let Some(v) = p.min_p {
                map.insert("min_p".into(), serde_json::json!(v));
            }
            if let Some(v) = p.frequency_penalty {
                map.insert("frequency_penalty".into(), serde_json::json!(v));
            }
            if let Some(v) = p.presence_penalty {
                map.insert("presence_penalty".into(), serde_json::json!(v));
            }
            if let Some(v) = p.repetition_penalty {
                map.insert("repetition_penalty".into(), serde_json::json!(v));
            }
            if let Some(v) = p.max_new_tokens {
                map.insert("max_new_tokens".into(), serde_json::json!(v));
            }
            if let Some(v) = p.min_new_tokens {
                map.insert("min_new_tokens".into(), serde_json::json!(v));
            }
            if !p.stop.is_empty() {
                map.insert("stop".into(), serde_json::json!(p.stop));
            }
            if !p.stop_token_ids.is_empty() {
                map.insert("stop_token_ids".into(), serde_json::json!(p.stop_token_ids));
            }
            if let Some(v) = p.ignore_eos {
                map.insert("ignore_eos".into(), serde_json::json!(v));
            }
            if let Some(v) = p.n {
                map.insert("n".into(), serde_json::json!(v));
            }
            if let Some(v) = p.seed {
                map.insert("sampling_seed".into(), serde_json::json!(v));
            }
            if p.guided_decoding.is_some() && (p.json_schema.is_some() || p.regex.is_some()) {
                return Err(
                    "legacy json_schema/regex cannot be combined with guided_decoding".into(),
                );
            }
            if let Some(guided) = p.guided_decoding.as_ref() {
                use proto::guided_decoding::Constraint;
                match guided.constraint.as_ref() {
                    Some(Constraint::JsonSchema(value)) if !value.is_empty() => {
                        map.insert("json_schema".into(), serde_json::json!(value));
                    }
                    Some(Constraint::Regex(value)) if !value.is_empty() => {
                        map.insert("regex".into(), serde_json::json!(value));
                    }
                    Some(Constraint::Ebnf(value)) if !value.is_empty() => {
                        map.insert("ebnf".into(), serde_json::json!(value));
                    }
                    Some(Constraint::Choice(choice))
                        if !choice.values.is_empty()
                            && choice.values.iter().all(|value| !value.is_empty()) =>
                    {
                        let alternatives = choice
                            .values
                            .iter()
                            .map(|value| regex_escape_literal(value))
                            .collect::<Vec<_>>()
                            .join("|");
                        map.insert(
                            "regex".into(),
                            serde_json::json!(format!("(?:{alternatives})")),
                        );
                    }
                    Some(Constraint::StructuralTag(value)) if !value.is_empty() => {
                        map.insert("structural_tag".into(), serde_json::json!(value));
                    }
                    Some(Constraint::Choice(_)) => {
                        return Err("guided choice must contain only non-empty values".into());
                    }
                    Some(_) => return Err("guided decoding constraint must not be empty".into()),
                    None => return Err("guided decoding constraint must be specified".into()),
                }
            } else {
                if let Some(value) = p.json_schema.as_ref() {
                    if value.is_empty() {
                        return Err("legacy json_schema must not be empty".into());
                    }
                    map.insert("json_schema".into(), serde_json::json!(value));
                }
                if let Some(value) = p.regex.as_ref() {
                    if value.is_empty() {
                        return Err("legacy regex must not be empty".into());
                    }
                    map.insert("regex".into(), serde_json::json!(value));
                }
            }
            Ok(serde_json::Value::Object(map))
        }
        None => Ok(serde_json::Value::Object(serde_json::Map::new())),
    }
}

fn insert_generation_controls(
    d: &mut HashMap<String, serde_json::Value>,
    priority: Option<i32>,
    require_reasoning: Option<bool>,
    max_thinking_tokens: Option<u32>,
) {
    if let Some(priority) = priority {
        d.insert("priority".into(), serde_json::json!(priority));
    }
    if let Some(require_reasoning) = require_reasoning {
        d.insert(
            "require_reasoning".into(),
            serde_json::json!(require_reasoning),
        );
    }
    if let Some(max_thinking_tokens) = max_thinking_tokens {
        d.insert(
            "max_thinking_tokens".into(),
            serde_json::json!(max_thinking_tokens),
        );
    }
}

fn trace_headers_to_json(headers: &HashMap<String, String>) -> Option<serde_json::Value> {
    if headers.is_empty() {
        None
    } else {
        Some(serde_json::json!(headers))
    }
}

fn insert_disaggregated_params(
    request: &mut HashMap<String, serde_json::Value>,
    params: &Option<proto::DisaggregatedParams>,
) {
    if let Some(params) = params {
        request.insert(
            "bootstrap_host".into(),
            serde_json::json!(params.bootstrap_host),
        );
        request.insert(
            "bootstrap_port".into(),
            serde_json::json!(params.bootstrap_port),
        );
        request.insert(
            "bootstrap_room".into(),
            serde_json::json!(params.bootstrap_room),
        );
    }
}

fn insert_router_hint(
    request: &mut HashMap<String, serde_json::Value>,
    hint: &Option<proto::RouterHint>,
) {
    let Some(hint) = hint else {
        return;
    };

    let mut value = serde_json::Map::new();
    if let Some(endpoint) = hint
        .source_control_endpoint
        .as_ref()
        .filter(|endpoint| !endpoint.is_empty())
    {
        value.insert(
            "source_control_endpoint".into(),
            serde_json::json!(endpoint),
        );
    }
    if !hint.block_hashes.is_empty() {
        value.insert("block_hashes".into(), serde_json::json!(hint.block_hashes));
    }

    let actions = hint
        .session_cache_actions
        .iter()
        .filter_map(|action| {
            if action.session_id.is_empty() {
                return None;
            }
            let cache_priority = match proto::SessionCachePriority::try_from(action.cache_priority)
            {
                Ok(proto::SessionCachePriority::Protected) => "protected",
                Ok(proto::SessionCachePriority::Evictable) => "evictable",
                _ => return None,
            };
            let mut action_value = serde_json::Map::new();
            action_value.insert("session_id".into(), serde_json::json!(action.session_id));
            action_value.insert("cache_priority".into(), serde_json::json!(cache_priority));
            if let Some(generation) = action.session_generation {
                action_value.insert("session_generation".into(), serde_json::json!(generation));
            }
            Some(serde_json::Value::Object(action_value))
        })
        .collect::<Vec<_>>();
    if !actions.is_empty() {
        value.insert(
            "session_cache_actions".into(),
            serde_json::Value::Array(actions),
        );
    }

    let demotions = hint
        .session_storage_demotions
        .iter()
        .filter_map(|demotion| {
            if demotion.operation_id.is_empty() || demotion.session_id.is_empty() {
                return None;
            }
            let mut demotion_value = serde_json::Map::new();
            demotion_value.insert(
                "operation_id".into(),
                serde_json::json!(demotion.operation_id),
            );
            demotion_value.insert("session_id".into(), serde_json::json!(demotion.session_id));
            if let Some(generation) = demotion.session_generation {
                demotion_value.insert("session_generation".into(), serde_json::json!(generation));
            }
            Some(serde_json::Value::Object(demotion_value))
        })
        .collect::<Vec<_>>();
    if !demotions.is_empty() {
        value.insert(
            "session_storage_demotions".into(),
            serde_json::Value::Array(demotions),
        );
    }
    if hint.prefetch_from_storage {
        value.insert("prefetch_from_storage".into(), serde_json::json!(true));
    }
    if hint.evict_session {
        value.insert("evict_session".into(), serde_json::json!(true));
    }

    if !value.is_empty() {
        request.insert("router_hint".into(), serde_json::Value::Object(value));
    }
}

fn now_timestamp() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

pub(crate) fn extract_model_path(json_info: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(json_info) {
        Ok(value) => value
            .get("model_path")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .unwrap_or_default(),
        Err(err) => {
            tracing::warn!("Failed to parse model info JSON: {}", err);
            String::new()
        }
    }
}

/// Build a request dict for GenerateReqInput from proto TextGenerateRequest fields.
pub(crate) fn build_text_generate_dict(
    rid: &str,
    req: &proto::TextGenerateRequest,
) -> Result<HashMap<String, serde_json::Value>, String> {
    let mut d = HashMap::new();
    d.insert("rid".into(), serde_json::json!(rid));
    d.insert("text".into(), serde_json::json!(req.text));
    d.insert(
        "sampling_params".into(),
        sampling_params_to_map(&req.sampling_params)?,
    );
    d.insert(
        "stream".into(),
        serde_json::json!(req.stream.unwrap_or(false)),
    );
    d.insert(
        "return_logprob".into(),
        serde_json::json!(req.return_logprob.unwrap_or(false)),
    );
    d.insert(
        "top_logprobs_num".into(),
        serde_json::json!(req.top_logprobs_num.unwrap_or(0)),
    );
    d.insert(
        "logprob_start_len".into(),
        serde_json::json!(req.logprob_start_len.unwrap_or(-1)),
    );
    d.insert(
        "return_text_in_logprobs".into(),
        serde_json::json!(req.return_text_in_logprobs.unwrap_or(false)),
    );
    if let Some(ref lp) = req.lora_path {
        d.insert("lora_path".into(), serde_json::json!(lp));
    }
    if let Some(ref rk) = req.routing_key {
        d.insert("routing_key".into(), serde_json::json!(rk));
    }
    if let Some(rank) = req.routed_dp_rank {
        d.insert("routed_dp_rank".into(), serde_json::json!(rank));
    }
    if let Some(ref session_id) = req.session_id {
        d.insert("session_id".into(), serde_json::json!(session_id));
    }
    insert_generation_controls(
        &mut d,
        req.priority,
        req.require_reasoning,
        req.max_thinking_tokens,
    );
    insert_disaggregated_params(&mut d, &req.disaggregated_params);
    insert_router_hint(&mut d, &req.router_hint);
    if let Some(trace) = trace_headers_to_json(&req.trace_headers) {
        d.insert("external_trace_header".into(), trace);
    }
    d.insert("received_time".into(), serde_json::json!(now_timestamp()));
    Ok(d)
}

/// Build a request dict for GenerateReqInput from proto GenerateRequest (tokenized).
pub(crate) fn build_generate_dict(
    rid: &str,
    req: &proto::GenerateRequest,
) -> Result<HashMap<String, serde_json::Value>, String> {
    let mut d = HashMap::new();
    d.insert("rid".into(), serde_json::json!(rid));
    d.insert("input_ids".into(), serde_json::json!(req.input_ids));
    d.insert(
        "sampling_params".into(),
        sampling_params_to_map(&req.sampling_params)?,
    );
    d.insert(
        "stream".into(),
        serde_json::json!(req.stream.unwrap_or(false)),
    );
    d.insert(
        "return_logprob".into(),
        serde_json::json!(req.return_logprob.unwrap_or(false)),
    );
    d.insert(
        "top_logprobs_num".into(),
        serde_json::json!(req.top_logprobs_num.unwrap_or(0)),
    );
    d.insert(
        "logprob_start_len".into(),
        serde_json::json!(req.logprob_start_len.unwrap_or(-1)),
    );
    if let Some(ref lp) = req.lora_path {
        d.insert("lora_path".into(), serde_json::json!(lp));
    }
    if let Some(ref rk) = req.routing_key {
        d.insert("routing_key".into(), serde_json::json!(rk));
    }
    if let Some(rank) = req.routed_dp_rank {
        d.insert("routed_dp_rank".into(), serde_json::json!(rank));
    }
    if let Some(ref session_id) = req.session_id {
        d.insert("session_id".into(), serde_json::json!(session_id));
    }
    insert_generation_controls(
        &mut d,
        req.priority,
        req.require_reasoning,
        req.max_thinking_tokens,
    );
    insert_disaggregated_params(&mut d, &req.disaggregated_params);
    insert_router_hint(&mut d, &req.router_hint);
    if let Some(trace) = trace_headers_to_json(&req.trace_headers) {
        d.insert("external_trace_header".into(), trace);
    }
    d.insert("received_time".into(), serde_json::json!(now_timestamp()));
    Ok(d)
}

/// Build a request dict for EmbeddingReqInput from proto TextEmbedRequest.
pub(crate) fn build_text_embed_dict(
    rid: &str,
    req: &proto::TextEmbedRequest,
) -> HashMap<String, serde_json::Value> {
    let mut d = HashMap::new();
    d.insert("rid".into(), serde_json::json!(rid));
    d.insert("text".into(), serde_json::json!(req.text));
    if let Some(ref rk) = req.routing_key {
        d.insert("routing_key".into(), serde_json::json!(rk));
    }
    if let Some(trace) = trace_headers_to_json(&req.trace_headers) {
        d.insert("external_trace_header".into(), trace);
    }
    d.insert("received_time".into(), serde_json::json!(now_timestamp()));
    d
}

/// Build a request dict for EmbeddingReqInput from proto EmbedRequest (tokenized).
pub(crate) fn build_embed_dict(
    rid: &str,
    req: &proto::EmbedRequest,
) -> HashMap<String, serde_json::Value> {
    let mut d = HashMap::new();
    d.insert("rid".into(), serde_json::json!(rid));
    d.insert("input_ids".into(), serde_json::json!(req.input_ids));
    if let Some(ref rk) = req.routing_key {
        d.insert("routing_key".into(), serde_json::json!(rk));
    }
    if let Some(trace) = trace_headers_to_json(&req.trace_headers) {
        d.insert("external_trace_header".into(), trace);
    }
    d.insert("received_time".into(), serde_json::json!(now_timestamp()));
    d
}

/// Build a request dict for EmbeddingReqInput from proto ClassifyRequest.
pub(crate) fn build_classify_dict(
    rid: &str,
    req: &proto::ClassifyRequest,
) -> HashMap<String, serde_json::Value> {
    let mut d = HashMap::new();
    d.insert("rid".into(), serde_json::json!(rid));
    if !req.text.is_empty() {
        d.insert("text".into(), serde_json::json!(req.text));
    }
    if !req.input_ids.is_empty() {
        d.insert("input_ids".into(), serde_json::json!(req.input_ids));
    }
    if let Some(ref rk) = req.routing_key {
        d.insert("routing_key".into(), serde_json::json!(rk));
    }
    if let Some(trace) = trace_headers_to_json(&req.trace_headers) {
        d.insert("external_trace_header".into(), trace);
    }
    d.insert("received_time".into(), serde_json::json!(now_timestamp()));
    d
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn generate_dicts_include_session_id() {
        let session_id = Some("session-1".to_string());
        let text_req = proto::TextGenerateRequest {
            session_id: session_id.clone(),
            ..Default::default()
        };
        let token_req = proto::GenerateRequest {
            session_id,
            ..Default::default()
        };

        assert_eq!(
            build_text_generate_dict("request-1", &text_req)
                .unwrap()
                .get("session_id"),
            Some(&serde_json::json!("session-1"))
        );
        assert_eq!(
            build_generate_dict("request-2", &token_req)
                .unwrap()
                .get("session_id"),
            Some(&serde_json::json!("session-1"))
        );
    }

    #[test]
    fn generate_dicts_include_typed_router_hint() {
        let router_hint = Some(proto::RouterHint {
            source_control_endpoint: Some("tcp://source:23280".to_string()),
            block_hashes: vec![11, 22],
            session_cache_actions: vec![
                proto::SessionCacheAction {
                    session_id: "session-a".to_string(),
                    cache_priority: proto::SessionCachePriority::Evictable as i32,
                    session_generation: Some(7),
                },
                proto::SessionCacheAction {
                    session_id: "session-b".to_string(),
                    cache_priority: proto::SessionCachePriority::Protected as i32,
                    session_generation: None,
                },
            ],
            session_storage_demotions: vec![proto::SessionStorageDemotion {
                operation_id: "demote-1".to_string(),
                session_id: "session-c".to_string(),
                session_generation: Some(9),
            }],
            prefetch_from_storage: true,
            evict_session: true,
        });
        let text_req = proto::TextGenerateRequest {
            router_hint: router_hint.clone(),
            ..Default::default()
        };
        let token_req = proto::GenerateRequest {
            router_hint,
            ..Default::default()
        };

        for mapped in [
            build_text_generate_dict("text-request", &text_req).unwrap(),
            build_generate_dict("token-request", &token_req).unwrap(),
        ] {
            assert_eq!(
                mapped["router_hint"],
                serde_json::json!({
                    "source_control_endpoint": "tcp://source:23280",
                    "block_hashes": [11, 22],
                    "session_cache_actions": [
                        {
                            "session_id": "session-a",
                            "cache_priority": "evictable",
                            "session_generation": 7,
                        },
                        {
                            "session_id": "session-b",
                            "cache_priority": "protected",
                        },
                    ],
                    "session_storage_demotions": [{
                        "operation_id": "demote-1",
                        "session_id": "session-c",
                        "session_generation": 9,
                    }],
                    "prefetch_from_storage": true,
                    "evict_session": true,
                })
            );
        }
    }

    #[test]
    fn generate_dicts_drop_invalid_router_actions() {
        let request = proto::GenerateRequest {
            router_hint: Some(proto::RouterHint {
                source_control_endpoint: None,
                block_hashes: Vec::new(),
                session_cache_actions: vec![proto::SessionCacheAction {
                    session_id: "session-a".to_string(),
                    cache_priority: proto::SessionCachePriority::Unspecified as i32,
                    session_generation: None,
                }],
                session_storage_demotions: Vec::new(),
                prefetch_from_storage: false,
                evict_session: false,
            }),
            ..Default::default()
        };

        assert!(
            !build_generate_dict("request", &request)
                .unwrap()
                .contains_key("router_hint")
        );
    }

    #[test]
    fn generate_dicts_include_disaggregated_params() {
        let disaggregated_params = Some(proto::DisaggregatedParams {
            bootstrap_host: "10.0.0.1".to_string(),
            bootstrap_port: 8998,
            bootstrap_room: i64::MAX,
        });
        let text_req = proto::TextGenerateRequest {
            disaggregated_params: disaggregated_params.clone(),
            ..Default::default()
        };
        let token_req = proto::GenerateRequest {
            disaggregated_params,
            ..Default::default()
        };

        for request in [
            build_text_generate_dict("request-1", &text_req),
            build_generate_dict("request-2", &token_req),
        ] {
            let request = request.unwrap();
            assert_eq!(
                request.get("bootstrap_host"),
                Some(&serde_json::json!("10.0.0.1"))
            );
            assert_eq!(
                request.get("bootstrap_port"),
                Some(&serde_json::json!(8998))
            );
            assert_eq!(
                request.get("bootstrap_room"),
                Some(&serde_json::json!(i64::MAX))
            );
        }
    }

    #[test]
    fn generate_dicts_omit_disaggregated_params_when_absent() {
        let text_request =
            build_text_generate_dict("request-1", &proto::TextGenerateRequest::default()).unwrap();
        let token_request =
            build_generate_dict("request-2", &proto::GenerateRequest::default()).unwrap();

        for request in [text_request, token_request] {
            assert!(!request.contains_key("bootstrap_host"));
            assert!(!request.contains_key("bootstrap_port"));
            assert!(!request.contains_key("bootstrap_room"));
        }
    }

    #[test]
    fn generate_dicts_preserve_optional_generation_controls() {
        let sampling_params = proto::SamplingParams {
            seed: Some(42),
            ..Default::default()
        };
        let text_request = proto::TextGenerateRequest {
            sampling_params: Some(sampling_params.clone()),
            priority: Some(3),
            require_reasoning: Some(false),
            max_thinking_tokens: Some(128),
            ..Default::default()
        };
        let token_request = proto::GenerateRequest {
            sampling_params: Some(proto::SamplingParams {
                seed: Some(42),
                ..Default::default()
            }),
            priority: Some(3),
            require_reasoning: Some(false),
            max_thinking_tokens: Some(128),
            ..Default::default()
        };

        for mapped in [
            build_text_generate_dict("text-request", &text_request).unwrap(),
            build_generate_dict("token-request", &token_request).unwrap(),
        ] {
            assert_eq!(mapped["priority"], serde_json::json!(3));
            assert_eq!(mapped["require_reasoning"], serde_json::json!(false));
            assert_eq!(mapped["max_thinking_tokens"], serde_json::json!(128));
            assert_eq!(
                mapped["sampling_params"]["sampling_seed"],
                serde_json::json!(42)
            );
        }

        for mapped in [
            build_text_generate_dict("text-request", &Default::default()).unwrap(),
            build_generate_dict("token-request", &Default::default()).unwrap(),
        ] {
            assert!(!mapped.contains_key("priority"));
            assert!(!mapped.contains_key("require_reasoning"));
            assert!(!mapped.contains_key("max_thinking_tokens"));
        }
    }

    #[test]
    fn guided_choice_maps_to_escaped_regex() {
        let request = proto::GenerateRequest {
            sampling_params: Some(proto::SamplingParams {
                guided_decoding: Some(proto::GuidedDecoding {
                    constraint: Some(proto::guided_decoding::Constraint::Choice(
                        proto::ChoiceConstraint {
                            values: vec!["a+b".into(), "x.y".into()],
                        },
                    )),
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mapped = build_generate_dict("request", &request).unwrap();
        assert_eq!(
            mapped["sampling_params"]["regex"],
            serde_json::json!("(?:a\\+b|x\\.y)")
        );
    }

    #[test]
    fn invalid_guidance_combinations_are_rejected() {
        let conflicting = proto::GenerateRequest {
            sampling_params: Some(proto::SamplingParams {
                regex: Some("[a-z]+".into()),
                guided_decoding: Some(proto::GuidedDecoding {
                    constraint: Some(proto::guided_decoding::Constraint::Regex("[0-9]+".into())),
                }),
                ..Default::default()
            }),
            ..Default::default()
        };

        let empty_choice = proto::GenerateRequest {
            sampling_params: Some(proto::SamplingParams {
                guided_decoding: Some(proto::GuidedDecoding {
                    constraint: Some(proto::guided_decoding::Constraint::Choice(
                        proto::ChoiceConstraint { values: vec![] },
                    )),
                }),
                ..Default::default()
            }),
            ..Default::default()
        };

        let empty_legacy_regex = proto::GenerateRequest {
            sampling_params: Some(proto::SamplingParams {
                regex: Some(String::new()),
                ..Default::default()
            }),
            ..Default::default()
        };

        for request in [conflicting, empty_choice, empty_legacy_regex] {
            assert!(build_generate_dict("request", &request).is_err());
        }
    }
}
