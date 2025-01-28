use crate::conversations::TokenizedInput;

/// python reference implementation
/// while i < limit:
// if curr_length == 0:
// curr_length+= data[i]["length"]
// bins.append(data[i])
// i+=1
// pbar.update(1)
// elif curr_length + data[i]["length"] <= target:
// bins[-1] = append_dict(bins[-1], data[i])
// curr_length+= data[i]["length"]
// i+=1
// pbar.update(1)
// else:
// curr_length = 0
// return Dataset.from_list(bins)
/// Takes in a sorted list of TokenizedInput and creates bins of TokenizedInput
pub fn create_bins(inputs: Vec<TokenizedInput>, max_length: u32) -> Vec<TokenizedInput> {
    let mut bins: Vec<TokenizedInput> = Vec::new();
    let mut curr_length = 0;
    let mut i = 0;
    while i < inputs.len() as u32 {
        if curr_length == 0 {
            curr_length += inputs[i as usize].length;
            bins.push(inputs[i as usize].clone());
            i += 1;
        } else if curr_length + inputs[i as usize].length <= max_length {
            bins.last_mut().unwrap().merge(&inputs[i as usize]);
            curr_length += inputs[i as usize].length;
            i += 1;
        } else {
            curr_length = 0;
        }
    }
    bins
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_bins() {
        let inputs = vec![
            TokenizedInput {
                input_ids: vec![4, 5, 6],
                labels: vec![4, 5, 6],
                position_ids: vec![0, 1, 2],
                length: 3,
            },
            TokenizedInput {
                input_ids: vec![7, 8, 9],
                labels: vec![7, 8, 9],
                position_ids: vec![0, 1, 2],
                length: 3,
            },
            TokenizedInput {
                input_ids: vec![0, 1],
                labels: vec![1, 2],
                position_ids: vec![0, 1],
                length: 2,
            },
        ];
        let max_length = 5;
        let bins = create_bins(inputs, max_length);
        assert_eq!(bins.len(), 2);
        assert_eq!(bins[0].length, 3);
        assert_eq!(bins[1].length, 5);
    }
}