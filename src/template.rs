// Much of this code was adapted from huggingface, for future reference
// https://github.com/huggingface/text-generation-inference/blob/main/router/src/infer/chat_template.rs
//
use crate::config::TokenizerConfig;
use minijinja::{Environment, Error, Template};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Default)]
pub(crate) struct ChatTemplateInputs<'a> {
    messages: Vec<TextMessage>,
    bos_token: Option<&'a str>,
    eos_token: Option<&'a str>,
    add_generation_prompt: bool,
}

#[derive(Clone, Deserialize, Serialize, Debug, PartialEq)]
pub struct TextMessage {
    pub role: String,
    pub content: String,
}
#[derive(Clone)]
pub struct ChatTemplate {
    template: Template<'static, 'static>,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl ChatTemplate {
    pub(crate) fn new(
        template: String,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Self {
        let env = Box::new(Environment::new());

        let template_str = template.into_boxed_str();

        // leaking env and template_str as read-only, static resources for performance.
        let template = Box::leak(env)
            .template_from_str(Box::leak(template_str))
            .unwrap();

        // check if the `tools` variable is used in the template

        Self {
            template,
            bos_token: bos_token.map(|token| token.as_str().to_string()),
            eos_token: eos_token.map(|token| token.as_str().to_string()),
        }
    }
    pub fn from_config(config: TokenizerConfig) -> Self {
        Self::new(
            config.chat_template,
            Some(config.bos_token),
            Some(config.eos_token),
        )
    }
    pub fn apply(&self, messages: Vec<TextMessage>) -> Result<String, Error> {
        self.template.render(ChatTemplateInputs {
            messages,
            bos_token: self.bos_token.as_deref(),
            eos_token: self.eos_token.as_deref(),
            add_generation_prompt: false,
        })
    }
}

// tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_template_new() {
        let template = ChatTemplate::new(
            "Hello, {{ name }}!".to_string(),
            Some("BOS".to_string()),
            Some("EOS".to_string()),
        );

        assert_eq!(template.bos_token, Some("BOS".to_string()));
        assert_eq!(template.eos_token, Some("EOS".to_string()));
    }
    #[test]
    fn test_apply_template() {
        let source = r#"{{- bos_token }}
        {%- if custom_tools is defined %}
            {%- set tools = custom_tools %}
        {%- endif %}
        {%- if not tools_in_user_message is defined %}
            {%- set tools_in_user_message = true %}
        {%- endif %}
        {%- if not date_string is defined %}
            {%- set date_string = "26 Jul 2024" %}
        {%- endif %}
        {%- if not tools is defined %}
            {%- set tools = none %}
        {%- endif %}

        {#- This block extracts the system message, so we can slot it into the right place. #}
        {%- if messages[0]['role'] == 'system' %}
            {%- set system_message = messages[0]['content']|trim %}
            {%- set messages = messages[1:] %}
        {%- else %}
            {%- set system_message = "" %}
        {%- endif %}

        {#- System message + builtin tools #}
        {{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
        {%- if builtin_tools is defined or tools is not none %}
            {{- "Environment: ipython\n" }}
        {%- endif %}
        {%- if builtin_tools is defined %}
            {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}}
        {%- endif %}
        {{- "Cutting Knowledge Date: December 2023\n" }}
        {{- "Today Date: " + date_string + "\n\n" }}
        {%- if tools is not none and not tools_in_user_message %}
            {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
            {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
            {{- "Do not use variables.\n\n" }}
            {%- for t in tools %}
                {{- t | tojson(indent=4) }}
                {{- "\n\n" }}
            {%- endfor %}
        {%- endif %}
        {{- system_message }}
        {{- "<|eot_id|>" }}

        {#- Custom tools are passed in a user message with some extra guidance #}
        {%- if tools_in_user_message and not tools is none %}
            {#- Extract the first user message so we can plug it in here #}
            {%- if messages | length != 0 %}
                {%- set first_user_message = messages[0]['content']|trim %}
                {%- set messages = messages[1:] %}
            {%- else %}
                {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
        {%- endif %}
            {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
            {{- "Given the following functions, please respond with a JSON for a function call " }}
            {{- "with its proper arguments that best answers the given prompt.\n\n" }}
            {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
            {{- "Do not use variables.\n\n" }}
            {%- for t in tools %}
                {{- t | tojson(indent=4) }}
                {{- "\n\n" }}
            {%- endfor %}
            {{- first_user_message + "<|eot_id|>"}}
        {%- endif %}

        {%- for message in messages %}
            {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
                {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
            {%- elif 'tool_calls' in message %}
                {%- if not message.tool_calls|length == 1 %}
                    {{- raise_exception("This model only supports single tool-calls at once!") }}
                {%- endif %}
                {%- set tool_call = message.tool_calls[0].function %}
                {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
                    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
                    {{- "<|python_tag|>" + tool_call.name + ".call(" }}
                    {%- for arg_name, arg_val in tool_call.arguments | items %}
                        {{- arg_name + '="' + arg_val + '"' }}
                        {%- if not loop.last %}
                            {{- ", " }}
                        {%- endif %}
                        {%- endfor %}
                    {{- ")" }}
                {%- else  %}
                    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
                    {{- '{"name": "' + tool_call.name + '", ' }}
                    {{- '"parameters": ' }}
                    {{- tool_call.arguments | tojson }}
                    {{- "}" }}
                {%- endif %}
                {%- if builtin_tools is defined %}
                    {#- This means we're in ipython mode #}
                    {{- "<|eom_id|>" }}
                {%- else %}
                    {{- "<|eot_id|>" }}
                {%- endif %}
            {%- elif message.role == "tool" or message.role == "ipython" %}
                {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
                {%- if message.content is mapping or message.content is iterable %}
                    {{- message.content | tojson }}
                {%- else %}
                    {{- message.content }}
                {%- endif %}
                {{- "<|eot_id|>" }}
            {%- endif %}
        {%- endfor %}
        {%- if add_generation_prompt %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
        {%- endif %}"#;
        // trim all the whitespace
        let source = source
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<&str>>()
            .join("");
        let env = Environment::new();
        let tmpl = env.template_from_str(&source);
        let chat_template_inputs = ChatTemplateInputs {
            messages: vec![
                TextMessage {
                    role: "user".to_string(),
                    content: "What is the capital of Singapore?".to_string(),
                },
                TextMessage {
                    role: "assistant".to_string(),
                    content: "I don't know, what is it?".to_string(),
                },
            ],
            bos_token: Some("<|begin_of_text|>"),
            eos_token: Some("<|eot_id|>"),
            add_generation_prompt: false,
            ..Default::default()
        };

        let result = tmpl.unwrap().render(chat_template_inputs).unwrap();

        assert_eq!(
            result,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of Singapore?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI don't know, what is it?<|eot_id|>"
        );
    }
    #[test]
    fn test_chat_template() {
        let env = Environment::new();

        let source = r#"
        {% for message in messages %}
            {% if message['role'] == 'system' %}
                {% if message['content']%}
                    {{'### System:\n' + message['content']+'\n\n'}}
                {% endif %}
            {% elif message['role'] == 'user' %}
                {{'### User:\n' + message['content']+'\n\n'}}
            {% elif message['role'] == 'assistant' %}
                {{'### Assistant:\n'  + message['content']}}
            {% endif %}
            {% if loop.last and add_generation_prompt %}
                {{ '### Assistant:\n' }}
            {% endif %}
        {% endfor %}"#;

        // trim all the whitespace
        let source = source
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<&str>>()
            .join("");

        let tmpl = env.template_from_str(&source);

        let chat_template_inputs = ChatTemplateInputs {
            messages: vec![
                TextMessage {
                    role: "user".to_string(),
                    content: "Hi!".to_string(),
                },
                TextMessage {
                    role: "assistant".to_string(),
                    content: "Hello how can I help?".to_string(),
                },
                TextMessage {
                    role: "user".to_string(),
                    content: "What is Deep Learning?".to_string(),
                },
                TextMessage {
                    role: "assistant".to_string(),
                    content: "magic!".to_string(),
                },
            ],
            bos_token: Some("[BOS]"),
            eos_token: Some("[EOS]"),
            add_generation_prompt: true,
            ..Default::default()
        };

        let result = tmpl.unwrap().render(chat_template_inputs).unwrap();

        assert_eq!(
            result,
            "### User:\nHi!\n\n### Assistant:\nHello how can I help?### User:\nWhat is Deep Learning?\n\n### Assistant:\nmagic!### Assistant:\n"
        );
    }
    #[test]
    fn test_with_apply() {
        let source = r#"
        {% for message in messages %}
            {% if message['role'] == 'system' %}
                {% if message['content']%}
                    {{'### System:\n' + message['content']+'\n\n'}}
                {% endif %}
            {% elif message['role'] == 'user' %}
                {{'### User:\n' + message['content']+'\n\n'}}
            {% elif message['role'] == 'assistant' %}
                {{'### Assistant:\n'  + message['content']}}
            {% endif %}
            {% if loop.last and add_generation_prompt %}
                {{ '### Assistant:\n' }}
            {% endif %}
        {% endfor %}"#;

        // trim all the whitespace
        let source = source
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<&str>>()
            .join("");

        let ct = ChatTemplate::new(
            source.to_string(),
            Some("[BOS]".to_string()),
            Some("[EOS]".to_string()),
        );

        let messages = vec![
            TextMessage {
                role: "user".to_string(),
                content: "Hi!".to_string(),
            },
            TextMessage {
                role: "assistant".to_string(),
                content: "Hello how can I help?".to_string(),
            },
            TextMessage {
                role: "user".to_string(),
                content: "What is Deep Learning?".to_string(),
            },
            TextMessage {
                role: "assistant".to_string(),
                content: "magic!".to_string(),
            },
        ];

        let result = ct.apply(messages).unwrap();

        assert_eq!(
            result,
            "### User:\nHi!\n\n### Assistant:\nHello how can I help?### User:\nWhat is Deep Learning?\n\n### Assistant:\nmagic!"
        );
    }
    #[test]
    fn test_with_tokenize() {
        let source = r#"
        {% for message in messages %}
            {% if message['role'] == 'system' %}
                {% if message['content']%}
                    {{'### System:\n' + message['content']+'\n\n'}}
                {% endif %}
            {% elif message['role'] == 'user' %}
                {{'### User:\n' + message['content']+'\n\n'}}
            {% elif message['role'] == 'assistant' %}
                {{'### Assistant:\n'  + message['content']}}
            {% endif %}
            {% if loop.last and add_generation_prompt %}
                {{ '### Assistant:\n' }}
            {% endif %}
        {% endfor %}"#;

        // trim all the whitespace
        let source = source
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<&str>>()
            .join("");

        let ct = ChatTemplate::new(
            source.to_string(),
            Some("[BOS]".to_string()),
            Some("[EOS]".to_string()),
        );

        let messages = vec![
            TextMessage {
                role: "user".to_string(),
                content: "Hi!".to_string(),
            },
            TextMessage {
                role: "assistant".to_string(),
                content: "Hello how can I help?".to_string(),
            },
            TextMessage {
                role: "user".to_string(),
                content: "What is Deep Learning?".to_string(),
            },
            TextMessage {
                role: "assistant".to_string(),
                content: "magic!".to_string(),
            },
        ];

        let result = ct.apply(messages).unwrap();

        assert_eq!(
            result,
            "### User:\nHi!\n\n### Assistant:\nHello how can I help?### User:\nWhat is Deep Learning?\n\n### Assistant:\nmagic!"
        );
    }
    #[test]
    fn test_with_config() {
        let config = TokenizerConfig {
            bos_token: "[BOS]".to_string(),
            eos_token: "[EOS]".to_string(),
            chat_template: "Test template".to_string(),
        };
        let ct = ChatTemplate::from_config(config);
        assert_eq!(ct.bos_token, Some("[BOS]".to_string()));
        assert_eq!(ct.eos_token, Some("[EOS]".to_string()));
        assert_eq!(ct.template.source(), "Test template");
    }
}
