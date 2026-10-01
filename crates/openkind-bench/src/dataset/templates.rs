//! Frozen request templates for the curated evaluation datasets.
//!
//! Every instruction string, criterion description, and rubric level below is
//! ported verbatim from the MIT-licensed `jev_benchmarking` task definitions
//! (`jev_benchmarking/tasks/*.py` at commit `6bbdeb33474849b6de2f0cccc9f5e19756abd67e`,
//! the harness published with arXiv 2609.37647) so OpenKind scores identical
//! requests and stays comparable with the paper's published numbers. Do not
//! edit the wording: a template change is a new template version.

/// SST-2 / movie-review sentiment criteria
/// (`tasks/classification.py`, `SENTIMENT_2`).
pub const SENTIMENT_2: [(&str, &str); 2] = [
    ("negative", "The text expresses a negative opinion."),
    ("positive", "The text expresses a positive opinion."),
];

/// SST-2 instruction (`tasks/classification.py`, `_Sentiment2.build`).
pub const SST2_INSTRUCTION: &str = "What is the overall sentiment of `review` toward the movie?";

/// AG News instruction and criteria (`tasks/classification.py`, `AGNews`).
pub const AG_NEWS_INSTRUCTION: &str = "What is the topic of the news `article`?";
pub const AG_NEWS_LABELS: [&str; 4] = ["World", "Sports", "Business", "Sci/Tech"];
pub const AG_NEWS_CRITERIA: [(&str, &str); 4] = [
    (
        "World",
        "International news, politics, conflicts and world affairs.",
    ),
    ("Sports", "Sports events, teams and athletes."),
    ("Business", "Companies, markets, the economy and finance."),
    (
        "Sci/Tech",
        "Science, technology, computing, the internet and space.",
    ),
];

/// Banking77 instruction (`tasks/classification.py`, `Banking77.build`).
pub const BANKING77_INSTRUCTION: &str = "Which intent best describes the bank customer's `query`?";

/// CLINC150 instruction and out-of-scope criterion
/// (`tasks/multiple_choice.py`, `CLINC150.build`).
pub const CLINC150_INSTRUCTION: &str =
    "Which intent does the user's `utterance` to a virtual assistant express?";
pub const CLINC150_OOS: (&str, &str) = (
    "oos",
    "Out of scope: the request matches none of the other intents.",
);

/// Shared multiple-choice instruction (`tasks/multiple_choice.py`,
/// `MC_INSTRUCTION`).
pub const MC_INSTRUCTION: &str = "Which option correctly answers `question`?";

/// HellaSwag instruction (`tasks/multiple_choice.py`, `HellaSwag.build`).
pub const HELLASWAG_INSTRUCTION: &str =
    "Which ending is the most plausible continuation of `context`?";

/// WinoGrande instruction (`tasks/multiple_choice.py`, `WinoGrande.build`).
pub const WINOGRANDE_INSTRUCTION: &str =
    "Which option correctly fills the blank `_` in `sentence`?";

/// BoolQ instruction (`tasks/multiple_choice.py`, `BoolQ.build`).
pub const BOOLQ_INSTRUCTION: &str = "According to `passage`, is the answer to `question` yes?";

/// PAWS instruction and criteria (`tasks/nli.py`, `PAWS.build`).
pub const PAWS_INSTRUCTION: &str = "Do `sentence_1` and `sentence_2` mean the same thing?";
pub const PAWS_CRITERIA: (&str, &str) = (
    "They are paraphrases: the same meaning, possibly with different wording or word order.",
    "Their meanings differ, even if they share most of their words.",
);

/// STS-B instruction and SemEval annotation levels (`tasks/scoring.py`, `STSB`).
pub const STSB_INSTRUCTION: &str = "How similar in meaning are `sentence_1` and `sentence_2`?";
pub const STSB_LEVELS: [&str; 6] = [
    "The two sentences are completely dissimilar.",
    "The two sentences are not equivalent, but are on the same topic.",
    "The two sentences are not equivalent, but share some details.",
    "The two sentences are roughly equivalent, but some important information differs or is missing.",
    "The two sentences are mostly equivalent, but some unimportant details differ.",
    "The two sentences are completely equivalent, as they mean the same thing.",
];

/// SST-5 instruction and levels (`tasks/scoring.py`, `SST5`).
pub const SST5_INSTRUCTION: &str = "What is the sentiment of `sentence`?";
pub const SST5_LEVELS: [&str; 5] = [
    "Very negative.",
    "Negative.",
    "Neutral.",
    "Positive.",
    "Very positive.",
];

/// The 151 CLINC150 intent names in ClassLabel index order, read from the
/// `clinc/clinc_oos` `plus` configuration feature metadata (`intent` is stored
/// as an integer in the converted parquet). `oos` is index 43.
pub const CLINC150_INTENTS: [&str; 151] = [
    "restaurant_reviews",
    "nutrition_info",
    "account_blocked",
    "oil_change_how",
    "time",
    "weather",
    "redeem_rewards",
    "interest_rate",
    "gas_type",
    "accept_reservations",
    "smart_home",
    "user_name",
    "report_lost_card",
    "repeat",
    "whisper_mode",
    "what_are_your_hobbies",
    "order",
    "jump_start",
    "schedule_meeting",
    "meeting_schedule",
    "freeze_account",
    "what_song",
    "meaning_of_life",
    "restaurant_reservation",
    "traffic",
    "make_call",
    "text",
    "bill_balance",
    "improve_credit_score",
    "change_language",
    "no",
    "measurement_conversion",
    "timer",
    "flip_coin",
    "do_you_have_pets",
    "balance",
    "tell_joke",
    "last_maintenance",
    "exchange_rate",
    "uber",
    "car_rental",
    "credit_limit",
    "oos",
    "shopping_list",
    "expiration_date",
    "routing",
    "meal_suggestion",
    "tire_change",
    "todo_list",
    "card_declined",
    "rewards_balance",
    "change_accent",
    "vaccines",
    "reminder_update",
    "food_last",
    "change_ai_name",
    "bill_due",
    "who_do_you_work_for",
    "share_location",
    "international_visa",
    "calendar",
    "translate",
    "carry_on",
    "book_flight",
    "insurance_change",
    "todo_list_update",
    "timezone",
    "cancel_reservation",
    "transactions",
    "credit_score",
    "report_fraud",
    "spending_history",
    "directions",
    "spelling",
    "insurance",
    "what_is_your_name",
    "reminder",
    "where_are_you_from",
    "distance",
    "payday",
    "flight_status",
    "find_phone",
    "greeting",
    "alarm",
    "order_status",
    "confirm_reservation",
    "cook_time",
    "damaged_card",
    "reset_settings",
    "pin_change",
    "replacement_card_duration",
    "new_card",
    "roll_dice",
    "income",
    "taxes",
    "date",
    "who_made_you",
    "pto_request",
    "tire_pressure",
    "how_old_are_you",
    "rollover_401k",
    "pto_request_status",
    "how_busy",
    "application_status",
    "recipe",
    "calendar_update",
    "play_music",
    "yes",
    "direct_deposit",
    "credit_limit_change",
    "gas",
    "pay_bill",
    "ingredients_list",
    "lost_luggage",
    "goodbye",
    "what_can_i_ask_you",
    "book_hotel",
    "are_you_a_bot",
    "next_song",
    "change_speed",
    "plug_type",
    "maybe",
    "w2",
    "oil_change_when",
    "thank_you",
    "shopping_list_update",
    "pto_balance",
    "order_checks",
    "travel_alert",
    "fun_fact",
    "sync_device",
    "schedule_maintenance",
    "apr",
    "transfer",
    "ingredient_substitution",
    "calories",
    "current_location",
    "international_fees",
    "calculator",
    "definition",
    "next_holiday",
    "update_playlist",
    "mpg",
    "min_payment",
    "change_user_name",
    "restaurant_suggestion",
    "travel_notification",
    "cancel",
    "pto_used",
    "travel_suggestion",
    "change_volume",
];

/// Option letters for multiple-choice criteria (`tasks/base.py`,
/// `option_keys`): A, B, C, ... The curated datasets need at most five.
pub fn option_keys(count: usize) -> Vec<String> {
    (0..count)
        .map(|index| char::from(b'A' + index as u8).to_string())
        .collect()
}
