# Project note

We start from the original **LeWorldModel** setup on **PushT**.

The original model predicts next latent states from:
- past latent states
- real action inputs

This project changes only one part:
- remove the real action inputs
- replace them with a small learned discrete codebook

The model should jointly:
- predict the next latent state
- infer a discrete code that provides the missing information needed for that prediction

The goal is to keep the rest of LeWM as unchanged as possible.

Initial plan:
1. reproduce the original PushT baseline
2. locate where actions enter the model
3. replace the action pathway with inferred discrete codes
4. compare against the original action-conditioned baseline
