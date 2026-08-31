# ADR 0001: App Runner with an evidence-gated file registry

Status: accepted for the portfolio deployment path

## Context

IntegrityBench needs one credible cloud path without turning a small moderation
service into a Kubernetes administration project. Model promotion must remain
separate from image deployment because a healthy container does not prove that
a model is safe.

## Decision

Use AWS App Runner for the stateless API, private versioned S3 for model and
registry objects, and DynamoDB for review metadata. Automatic App Runner image
deployment is disabled. The application loads only the registry's production
model and verifies its SHA-256. Registry promotion separately requires an
approved evaluation artifact.

## Consequences

This keeps the deployment small and gives the model lifecycle an explicit
boundary. It does not provide weighted traffic splitting, so canary instances
are separate services evaluated by a real shadow sample before promotion. If
traffic, networking, or rollout needs become more complex, revisit ECS or EKS.
