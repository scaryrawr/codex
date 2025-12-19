//! Conversion utilities for OSS model presets.

use codex_protocol::openai_models::ModelPreset;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::openai_models::ReasoningEffortPreset;

/// Convert raw model ID strings into ModelPreset structs.
pub(crate) fn to_presets(model_ids: Vec<String>, provider: &str) -> Vec<ModelPreset> {
    model_ids
        .into_iter()
        .enumerate()
        .map(|(idx, model_id)| ModelPreset {
            id: model_id.clone(),
            model: model_id.clone(),
            display_name: model_id,
            description: format!("{provider} model"),
            default_reasoning_effort: ReasoningEffort::None,
            // OSS models don't support reasoning effort configuration.
            // We include a single "None" entry so the UI dismisses on select
            // (the UI checks `supported_reasoning_efforts.len() == 1` for this).
            supported_reasoning_efforts: vec![ReasoningEffortPreset {
                effort: ReasoningEffort::None,
                description: "Default".to_string(),
            }],
            is_default: idx == 0,
            upgrade: None,
            show_in_picker: true,
            supported_in_api: true,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn to_presets_sets_first_as_default() {
        let ids = vec!["model-a".into(), "model-b".into()];
        let presets = to_presets(ids, "Test");

        assert_eq!(presets.len(), 2);
        assert!(presets[0].is_default);
        assert!(!presets[1].is_default);
        assert_eq!(presets[0].model, "model-a");
        assert_eq!(presets[0].description, "Test model");
    }

    #[test]
    fn to_presets_empty_input() {
        let presets = to_presets(vec![], "Test");
        assert!(presets.is_empty());
    }

    #[test]
    fn to_presets_single_model_is_default() {
        let ids = vec!["only-model".into()];
        let presets = to_presets(ids, "Provider");

        assert_eq!(presets.len(), 1);
        assert!(presets[0].is_default);
        assert_eq!(presets[0].model, "only-model");
        assert_eq!(presets[0].display_name, "only-model");
    }
}
