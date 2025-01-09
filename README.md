# Adaptive Attention with Guided Prompts

This section describes the model architecture leveraging the **Adaptive Attention Method** for generating personalized feedback. The architecture consists of two custom encoders one for the context encoding and the other for the metacognitive embedding layer and a decoder.

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
- **Custom Transformer Block**: Comprises multiple layers(4) to handle attention mechanisms which takes input as a combined input of a guided prompt with problem+student_code combination which is by the way the student's initial state of solution.

### Custom Transformer Block
The custom transformer block takes the above combined input, the student persona from the metacognitive embedding layer, and the context encoded problem context  and outputs each next state of the student's solution.

Each transformer block contains:
1. **Self-Attention Layer \( hR \)**: Calculates the self-attention of the student's current state (initial and subsequent states).
2. **Cross-Attention Layer 1 \( oP \)**: Computes cross-attention between the student persona and student code.
3. **Cross-Attention Layer 2 \( oC \)**: Computes cross-attention between the problem context and student code.
4. **PAA Layer Output \( HPAA \)**: Processes the attention outputs through the PAA Layer to unify persona and context.
5. **MLP/RNN Layer**: Processes the \( HPAA \) output to generate the transformer block output.

The above transformer blocks are repeated 4 times to refine the output, which is then passed through a **final fully connected layer** to generate the final logits.

## PAA Layer (Personalized Attention Adaptation Layer)

The **PAA Layer** integrates the student's cognitive profile with the problem-solving context to provide personalized feedback. It takes three inputs: **\( hR \), \( oP \), and \( oC \),** and processes them through the following steps:

1. **Persona Importance Score (\( Mp \))**:

   ![Equation](https://latex.codecogs.com/svg.latex?Mp%20=%20\sigma(\text{fc}(\text{concat}[hR,%20oP],%20\text{dim}=-1)))

   **Explanation**: Combines the student's profile (\( oP \)) with their problem-solving context (\( hR \)), determining how much of the student's persona should influence the feedback. The learnable fully connected layer (fc) and sigmoid function allow the model to adjust the weight dynamically.
   Here the hR and oP are concatenated such that it combines the student's profile with the student's problem-solved context. Since this persona+ student's code depends on each student's unique metacognitive profiles, the sigmoid function with fc(learnable) it self learns how much proportion/weight and what aspects of the student's profile need to be considered for feedback/output.


3. **Context Importance Score (\( Mc \))**:
   
   ![Equation](https://latex.codecogs.com/svg.latex?Mc%20=%201%20-%20Mp)

   **Explanation**: Complements the persona weight, balancing attention between the student’s cognitive state and the correctness of the solution. This ensures feedback can emphasize areas that require improvement, such as weaker metacognitive aspects.
   Takes the complement of the persona weight. Amount of attention paid to the student's metacognitive state vs the correctness of the solution. i.e. if the student has a high metacognitive profile more proportion is weighted on that. ( I suppose this needs to be changed to the other way around, but  not sure , since we need to consider improving each student's lower metacognitive aspects.)

5. **Weighted Outputs**:
   
   ![Equation](https://latex.codecogs.com/svg.latex?oP^{\text{weighted}}%20=%20Mp%20\odot%20oP)


   ![Equation](https://latex.codecogs.com/svg.latex?oC^{\text{weighted}}%20=%20Mc%20\odot%20oC)

   **Explanation**: Applies the computed weights to the persona (\( oP \)) and context (\( oC \)) cross-attention outputs, tailoring the feedback according to the student's needs. If the student shows strong metacognitive skills, the feedback will emphasize persona aspects, whereas weaker cognitive profiles will receive more context-based feedback.
   Element-wise multiplication/dot product): Assign the above relative importances/weights to the cross attention outputs (more metacognitive weight <---- feedback sided with more persona and student struggling with code get with low metacognitive weights more context on code/solution. --> So here the personalized feedback can be either be personalized as up to student's strong metacognitive profile or as upto the student's weak metacognitive profile with more coding debug according to student's solution.

7. **Combined Output (\( HPAA \))**:

    ![Equation](https://latex.codecogs.com/svg.latex?HPAA%20=%20oP^{\text{weighted}}%20+%20oC^{\text{weighted}})

   **Explanation**: Integrates the weighted persona and context information, producing a unified representation that combines the student's profile with the problem context.
   Unifies above both information, thereby the student's profile can blend with the problem context which is calculated towards the student's code.

9. **Final Output**:

   ![Equation](https://latex.codecogs.com/svg.latex?\text{Output}%20=%20\text{fc}(HPAA))

   **Explanation**: A final fully connected layer maps the combined output to the task-specific output space, which could be a sequence of personalized feedback or other relevant outputs.
   fc(learnable) maps the HPAA into the outer space, which is an output necessary for the task a sequence of personalized feedback. 

   ## Equations

### Mp Equation

![Equation](https://latex.codecogs.com/svg.latex?Mp%20=%20\sigma(\text{fc}(\text{concat}[hR,%20oP],%20\text{dim}=-1)))

### Mc Equation

![Equation](https://latex.codecogs.com/svg.latex?Mc%20=%201%20-%20Mp)

### Weighted Persona Output

![Equation](https://latex.codecogs.com/svg.latex?oP^{\text{weighted}}%20=%20Mp%20\odot%20oP)

### Weighted Context Output

![Equation](https://latex.codecogs.com/svg.latex?oC^{\text{weighted}}%20=%20Mc%20\odot%20oC)

### Combined Output (HPAA)

![Equation](https://latex.codecogs.com/svg.latex?HPAA%20=%20oP^{\text{weighted}}%20+%20oC^{\text{weighted}})

### Final Output

 ![Equation](https://latex.codecogs.com/svg.latex?\text{Output}%20=%20\text{fc}(HPAA))

## Conclusion

This model architecture enables the generation of personalized feedback by adaptively balancing attention between a student’s cognitive profile and the problem context. By leveraging the PAA Layer and guided prompts, the model provides context-aware and persona-specific guidance, enhancing the learning experience for each student.
