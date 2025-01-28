use serde::{Deserialize, Serialize};
use std::fs;

use crate::template;

#[derive(Debug, Serialize, Deserialize)]
struct __Conversation {
    conversation: Vec<template::TextMessage>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Conversation {
    conversation: Vec<String>,
}

#[derive(Clone, Serialize)]
pub struct TokenizedInput {
    pub input_ids: Vec<u32>,
    pub labels: Vec<i32>,
    pub position_ids: Vec<u32>,
    pub length : u32,
}

impl TokenizedInput {
    pub fn merge(&mut self, other: &TokenizedInput) {
        // Change the method to take &mut self instead of &self
        self.input_ids.extend(other.input_ids.clone());
        self.labels.extend(other.labels.clone());
        self.position_ids.extend(other.position_ids.clone());
        self.length += other.length;
    }
}

pub fn read_jsonl(jsonl_path: &str, ct: template::ChatTemplate) -> Vec<String> {
    let jsonl = fs::read_to_string(jsonl_path).unwrap();
    let mut conversations: Vec<String> = Vec::new();
    for line in jsonl.lines() {
        let conv: __Conversation = serde_json::from_str(line).unwrap();
        let result = ct.apply(conv.conversation).unwrap();
        conversations.push(result);
    }
    conversations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge() {
        let mut left = TokenizedInput {
            input_ids: vec![1, 2, 3],
            labels: vec![1, 2, 3],
            position_ids: vec![0, 1, 2],
            length: 3,
        };
        let right = TokenizedInput {
            input_ids: vec![4, 5, 6],
            labels: vec![4, 5, 6],
            position_ids: vec![0, 1, 2],
            length: 3,
        };
        left.merge(&right);
        assert_eq!(left.input_ids, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(left.labels, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(left.position_ids, vec![0, 1, 2, 0, 1, 2]);
        assert_eq!(left.length, 6);
    }
}