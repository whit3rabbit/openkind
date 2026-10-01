# SDK Test Parity Map — typesafe-sdk-python → openkind-client

Every test file in the TypeSafe Python SDK (`github.com/typesafe-ai/typesafe-sdk-python`, `tests/`), mapped to its Rust counterpart. Reference: SDK v0.7.2 (commit `f078f1e208a0d885154dc758344ae4fce77ac168`); clone that commit into a temporary directory (e.g. `$TMPDIR` or `%TEMP%`) when re-auditing.

Status legend: **ported** (assertions live in Rust), **covered** (equivalent guarantee via a different mechanism), **N/A** (Python/platform-specific, reason given), **divergence** (deliberate behavioral difference, pinned by a test).

## Retry — `tests/test_retry.py` → `tests/sdk_parity_retry.rs` + unit tests in `src/retry/tests.rs`

| Python test | Rust port |
|---|---|
| `test_retry_policy_invalid_timeout` | covered — `Duration` cannot be negative/NaN; `timeout(0)` rejected at build (`sdk_parity_config::invalid_settings_rejected_at_build`, unit `zero_timeout_rejected_at_build`) |
| `test_zero_backoff_retries` | ported — `zero_backoff_still_retries` (both outcomes, retry-count `[None, "1"]`) |
| `test_invalid_backoff` / `test_invalid_backoff_jitter` / `test_invalid_max_retries` | covered — jitter validated at build; durations/u32 make the rest unrepresentable |
| `test_retry_policy_timeout_budget` | ported — `budget_stops_retrying`, `budget_is_fresh_per_call` (real clock, ms-scale) |
| `test_retry_policy_timeout_override` | ported — per-call `RequestOptions::retry`/`timeout` (`per_call_*`, `timeout_errors_are_retried_then_recover`) |
| `test_default_retry_statuses` | ported — `default_retry_statuses_match_python_matrix` (13-status table incl. 529) + `retried_requests_carry_python_retry_count_sequence` |
| `test_connection_retry_recovers` | ported (timeout flavor) — `timeout_errors_are_retried_then_recover`; connection-refused flavor in `tests/retry_behavior.rs::connection_error_is_classified_and_retried` |
| `test_server_delay_through_tenacity` | ported — `retry_after_ms_honored_over_long_backoff`, `retry_after_raw_seconds_honored` (elapsed-bounds) |
| `test_parse_retry_after` | ported — unit `retry_after_python_sdk_parity_matrix` (empty→0, NaN/-1/inf, overflow→None, ms-fallthrough) |
| `test_backoff_dates_cap_and_jitter` | ported — unit `retry_after_http_date*`, `long_server_delays_are_honored_without_cap`, backoff progression/jitter units |
| `test_system_one_retry_override` | ported — `per_call_retry_override_wins`, `per_call_retry_override_can_extend_retries` |
| `test_async_concurrent_retry_state` | ported — `concurrent_calls_have_independent_retry_state` |
| `test_system_one_retry_recovers_with_overrides` | ported — `retry_recovery_with_full_wire_assertions` (per-attempt body/header asserts, default-model follow-up) |
| `test_concurrent_system_one_overrides` | ported — `concurrent_calls_with_distinct_policies_and_models` |
| `test_exhausted_transport_retry` | ported/covered — timeout classification (`transport_errors_classify_like_python`), connection variant (retry_behavior.rs) |
| `test_exhausted_retry_preserves_final_http_error` | ported — `exhausted_retry_preserves_final_http_error` (last response wins; exact Display) |
| `test_cancel_pending_retry` | ported — `cancel_pending_retry` (tokio abort during server-requested sleep) |
| `test_retry_policy_max_retries` | ported — `max_retries_table` (0/1/4 → 1/2/5 attempts) |
| `test_retry_policy_custom_statuses` | ported — `custom_statuses_replace_defaults` (unit + behavioral) via `with_retry_statuses` + `retry_server_errors(false)` |
| `test_retry_policy_per_call_override` | ported — see above |
| `test_retry_policy_exceptions_and_predicate` | ported — `retry_predicate_opts_404_into_retries` via `retry_predicate` |
| `test_retry_policy_wait_options` | covered — unit `retry_after_takes_precedence` (`respect_retry_after(false)` → backoff) |
| `test_backoff_extreme_values` | ported — unit `backoff_extreme_values` (nanosecond precision, saturation) |

## Errors — `tests/test_errors.py` + `test_clients.py` error sections → `tests/sdk_parity_errors.rs`

| Python test | Rust port |
|---|---|
| `test_exception_reconstruction` | ported — `errors_are_send_sync_across_threads` |
| `test_api_error_from_process_pool` | covered — same Send/Sync guarantee |
| `test_api_error_request_context` | ported — `api_error_endpoint_names_resource_without_credentials` |
| `test_api_error_endpoint_omits_url_credentials` | ported — same test (key absent from Display/endpoint) |
| `test_message_override` | N/A — Rust `ApiError` fields are public; construct directly |
| `test_error_body_edge_cases` | ported — `error_body_edge_cases_do_not_panic` (empty, non-UTF8) + `non_json_error_bodies_fall_back_like_python` |
| `test_error_mapping` | ported — `error_mapping_matches_python_exception_taxonomy` (11 statuses, request id, retry-after-ms=125, exact Display) |
| `test_error_messages` | ported — `error_messages_match_python_extraction_matrix` (priority order incl. `detail.message`, FastAPI detail list) + raw-text fallback |
| `test_transport_errors` | ported — `transport_errors_classify_like_python` (timeout with value / connection) |
| `test_headers_timeout_and_logging` | ported (header half) — `protected_headers_cannot_be_overridden`; log-redaction half is N/A (no log-capture harness; secrets never enter `Debug` — unit `client_debug_does_not_leak_api_key`) |

## Clients / wire — `tests/test_clients.py`, `test_types.py`, `test_responses.py`, `test_questions.py` → `tests/sdk_parity_wire.rs`, `tests/live_server.rs`

| Python test | Rust port |
|---|---|
| `test_round_trip` (dataclass/raw/mixed) | ported — `round_trip_exact_wire_body_and_typed_response` (typed construction is the only form; raw-dict passthrough is unrepresentable) |
| `test_extra_body_shallow_override` | N/A — the Rust API takes a typed `SystemRequest` (no `extra_body` parameter) |
| `test_unserializable_request_body_raises` | N/A — typed structs always serialize |
| `test_raw_question_passthrough` | covered — typed structs are the passthrough; wire asserts in round trip |
| `test_question_schema_validation_is_left_to_api` | ported — `tests/live_server.rs::question_schema_validation_is_left_to_api` (empty score/choice criteria, empty instructions → 422 `invalid_body`) |
| `test_rich_descriptions` | ported — `rich_instructions_wire_through` |
| `test_models_shape` + `test_models_ignore_unknown_fields` | ported — `models_response_shape_and_unknown_fields` |
| `test_invalid_models_response` | ported — `invalid_models_response_is_decode_error` (4 shapes) |
| `test_validation_before_network` | divergence — typed construction makes structurally-invalid questions unrepresentable; semantic validation is server-side (422), see above |
| `test_system_one_timeout_override` | covered — per-call timeout exercised in `transport_errors_classify_like_python` and the retry suite; reqwest cannot expose the effective per-request timeout for introspection |
| `test_http_client_settings` + `test_http_client_timeout_precedence` | ported — `supplied_http_client_still_gets_sdk_timeout` |
| `test_supplied_network_resources_closed` / `test_owned_http_client_closed` / `test_exceptional_context_closes_http_client` | N/A — reqwest clients have no close protocol; connection pools clean up on drop |
| `test_task_cancellation_closes_context` / `test_cancellation_propagates` | ported (cancellation half) — `cancel_pending_retry`; pool-close half N/A as above |
| `test_types.py` array/object/None states | ported — `array_and_object_states_wire_through`; `None`-state N/A (no optional state in the wire schema) |
| `test_str_subclasses_fallback_to_strings` | N/A — Rust has no str subclassing |
| `test_responses.py` malformed bodies | ported — `malformed_responses_are_decode_errors` |
| `test_nested_missing_field_path` | divergence — `invalid_models_response_is_decode_error` pins the user-visible field name (`missing field \`name\``); serde errors do not carry the `models[1]` index path |
| `test_response_carries_request_id` | divergence — success responses decode to plain data structs; the request id is surfaced on errors via `Error::request_id` |
| `test_response_carries_raw_http_response` / `test_response_serialization_excludes_http_metadata` / `test_copied_response_preserves_metadata` / `test_missing_raw_raises_on_access` / `test_missing_request_id_raises_on_access` | N/A — no raw-HTTP-response wrapper; decoded structs carry data only |
| `test_responses.py` unknown extra fields | ported — `unknown_extra_fields_tolerated` |
| `test_public_response_types_ignore_unknown_fields` | ported — `unknown_extra_fields_tolerated` |
| `test_responses.py` unknown answer type ignored | divergence — `unknown_answer_type_is_strict` (fail closed instead of dropping judgments) |
| `test_responses.py` answer/request correspondence | divergence — `response_validation.rs` rejects a decoded 2xx response with missing or extra answer IDs, a mismatched answer type, or an out-of-list Choice; transport-only stubs return answers matching the submitted questions |
| `test_response_preserves_nested_json` | divergence — the SDK's `JSONContent` typing admits arbitrary JSON in legend entries; openkind pins the live-OpenAPI shape (string labels), covered by the round trip |
| `test_responses.py` typed attributes / frozen / cached groups | ported/covered — typed matches in round trip; mutability/caching N/A (plain data structs) |
| `test_questions.py` discriminators/omitted defaults/reserved keys | ported — `question_discriminators_and_omitted_defaults` + `src/question.rs` unit tests |
| `test_normalization_preserves_objects` | ported — `question_discriminators_and_omitted_defaults` (same wire JSON: omitted noul criteria, `null`-valued choice criteria, score criteria arrays) |
| `test_normalization_preserves_raw_questions` / `test_raw_questions_require_structural_keys` | covered — no raw-dict path exists; typed structs are the passthrough (see `test_raw_question_passthrough`), and the wire deserializer tolerates unknown question fields like the SDK (`IgnoredAny` in `openkind-core/src/question.rs`) |
| `test_optional_noul_criteria` | ported — `question_discriminators_and_omitted_defaults` (criteria omitted unless set; reserved `true`/`false` keys when set) |
| `test_covariant_question_mappings` | N/A — mapping covariance is a Python typing property; `HashMap<String, Question>` is the fixed surface |
| `test_questions.py` construction-time validation | divergence — Rust structs don't validate at construction (server 422s; see `question_schema_validation_is_left_to_api`) |

## Config — `tests/test_config.py` → `tests/sdk_parity_config.rs` + unit tests in `src/client/tests.rs`

| Python test | Rust port |
|---|---|
| `test_transport_and_http_client_mutually_exclusive` | N/A — no transport abstraction; `http_client` escape hatch only |
| `test_model_override` | ported — `model_override_beats_client_default` |
| `test_resolution` | ported — unit `resolve_lookup_precedence` (explicit → `OPENKIND_*` → `TYPESAFE_*`) |
| `test_missing_key` | ported — `explicit_empty_env_names_every_variable_in_error` + `tests/live_server.rs::missing_api_key_fails_at_build_time` |
| `test_api_key_whitespace` | ported — unit `api_key_resolution_matches_python_sdk` (all four paddings, both sources) + `padded_builder_api_keys_are_trimmed_before_the_wire` (`Bearer test-key` asserted on the wire) |
| `test_invalid_explicit_key_does_not_fall_back_to_env` | ported — unit `api_key_resolution_matches_python_sdk` (explicit empty/NUL keys error even with a valid env key present) |
| `test_invalid_api_key` | ported — unit `api_key_resolution_matches_python_sdk` (8 characters × both sources; the error text never echoes the credential) |
| `test_empty_env_unset` | ported — unit `clean_env_value` assertions |
| `test_invalid_timeout` | ported — `invalid_settings_rejected_at_build` |
| `test_timeout_object` / per-dimension timeouts | N/A — reqwest exposes a single per-request `Duration` |

## N/A wholesale (Python tooling/platform)

- `test_docs.py`, `test_release_notes.py` — docstring/changelog linting.
- `test_typing.py`, `tests/typing/` — mypy negative typing; Rust's type system covers this at compile time.
- `test_logging.py` — `logging`-module plumbing and log-filter redaction. The Rust client never logs headers or bodies (only `tracing` events without credentials), and `client_debug_does_not_leak_api_key` pins the `Debug` surface.
- `test_public_api_surface.py` — import-surface mechanics; covered by `sdk_constants_match_python_defaults`, `request_builders_compose_without_network`, and doc tests.
- `test_public_sync.py` — SDK-repo release tooling (signing, snapshotting, GitHub push); not client behavior.
- `test_pydantic_response_models.py`, `test_integration.py` (live `api.typesafe.ai` calls requiring a real key) — the live-server suite runs the same contracts against a real local `openkindd` router instead.
