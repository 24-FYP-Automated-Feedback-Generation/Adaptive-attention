# Adaptive Attention with Guided Prompts

This section describes the model architecture leveraging the **Adaptive Attention Method** for generating personalized feedback. The architecture consists of two custom encoders and a decoder, each playing a crucial role in integrating context and metacognitive information to provide tailored feedback.

## Model Architecture Overview

### Context Encoder
- **Purpose**: Encodes the problem statement and expected correct code, concatenated with the prompt.
- **Output**: A vector of dimension `(4, 1024)`.

### Metacognitive Embedding Layer
- **Purpose**: Encodes the student’s metacognitive profile, a 16-dimensional vector, concatenated with a metacognition-incorporated prompt.
- **Output**: A vector of dimension `(4, 1024)`.

### Decoder
The decoder processes the combined input of the guided prompt and the student's initial solution state (problem + student code) through the following components:
- **Token Embedding Layer**: Embeds the input tokens.
- **Positional Embedding Layer**: Adds positional information to the embeddings.
- **Custom Transformer Block**: Comprises multiple layers to handle attention mechanisms and sequence generation.

### Custom Transformer Block
Each transformer block contains:
1. **Self-Attention Layer \( hR \)**: Calculates the self-attention of the student's current state (initial and subsequent states).
2. **Cross-Attention Layer 1 \( oP \)**: Computes cross-attention between the student persona and student code.
3. **Cross-Attention Layer 2 \( oC \)**: Computes cross-attention between the problem context and student code.
4. **PAA Layer Output \( HPAA \)**: Processes the attention outputs through the PAA Layer to blend persona and context.
5. **MLP/RNN Layer**: Processes the \( HPAA \) output to generate the transformer block output.

The above transformer blocks are repeated four times to refine the output, which is then passed through a fully connected layer to generate the final logits.

## PAA Layer (Personalized Attention Allocation Layer)

The **PAA Layer** integrates the student's cognitive profile with the problem-solving context to provide personalized feedback. It takes three inputs: \( hR \), \( oP \), and \( oC \), and processes them through the following steps:

1. **Persona Importance Score (\( Mp \))**:
   $
   M_P = \sigma(\text{fc}(\text{concat}[h_R, o_P], \text{dim}=-1))
   $
   **Explanation**: Combines the student's profile (\( oP \)) with their problem-solving context (\( hR \)), determining how much of the student's persona should influence the feedback. The learnable fully connected layer (fc) and sigmoid function allow the model to adjust the weight dynamically.

2. **Context Importance Score (\( Mc \))**:
   $
   Mc = 1 - Mp
   $
   **Explanation**: Complements the persona weight, balancing attention between the student’s cognitive state and the correctness of the solution. This ensures feedback can emphasize areas that require improvement, such as weaker metacognitive aspects.

3. **Weighted Outputs**:
   $
   oP^{\text{weighted}} = Mp \odot oP
   $
   $
   oC^{\text{weighted}} = Mc \odot oC
   $
   **Explanation**: Applies the computed weights to the persona (\( oP \)) and context (\( oC \)) cross-attention outputs, tailoring the feedback according to the student's needs. If the student shows strong metacognitive skills, the feedback will emphasize persona aspects, whereas weaker cognitive profiles will receive more context-based feedback.

4. **Combined Output (\( HPAA \))**:
   $
   HPAA = oP^{\text{weighted}} + oC^{\text{weighted}}
   $
   **Explanation**: Integrates the weighted persona and context information, producing a unified representation that combines the student's profile with the problem context.

5. **Final Output**:
   $
   \text{Output} = \text{fc}(HPAA)
   $
   **Explanation**: A final fully connected layer maps the combined output to the task-specific output space, which could be a sequence of personalized feedback or other relevant outputs.

## Conclusion

This model architecture enables the generation of personalized feedback by adaptively balancing attention between a student’s cognitive profile and the problem context. By leveraging the PAA Layer and guided prompts, the model provides context-aware and persona-specific guidance, enhancing the learning experience for each student.
