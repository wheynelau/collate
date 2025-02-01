use crate::conversations::TokenizedInput;
use rayon::prelude::*;
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
pub fn create_bins(mut inputs: Vec<TokenizedInput>, max_length: u32) -> Vec<TokenizedInput> {
    
    let normalized_inputs: Vec<TokenizedInput> = inputs
        .par_chunks_mut(1_000_000usize)
        .flat_map(|chunk| {
            let mut current_bin = Vec::new();
            let mut current_length = 0;
            let mut bin = None;

            for input in chunk.iter() {
                let input_length = input.length;
                
                if current_length == 0 {
                    if input_length > max_length {
                        continue;
                    }
                    bin = Some(input.clone());
                    current_length = input_length;
                } else if current_length + input_length <= max_length {
                    if let Some(ref mut b) = bin {
                        b.merge(input);
                    }
                    current_length += input_length;
                } else {
                    if let Some(b) = bin.take() {
                        current_bin.push(b);
                    }
                    bin = Some(input.clone());
                    current_length = input_length;
                }
            }
            
            if let Some(b) = bin {
                current_bin.push(b);
            }
            
            current_bin
        })
        .collect();

    normalized_inputs
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