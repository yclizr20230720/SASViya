# Wafer Defect Analysis System - Detailed Design Specification

**Document Version:** 2.0

**Author:** Manus AI

**Date:** January 17, 2026

**Status:** Final

---

## 1. Introduction

This document provides the detailed technical design and architecture for the AI-driven Wafer Defect Analysis System. It serves as the engineering blueprint for development, translating the functional and non-functional requirements into concrete implementation details. The design prioritizes scalability, resilience, maintainability, and performance to meet the demanding needs of a high-volume semiconductor manufacturing environment.

All diagrams within this document are rendered using Mermaid syntax to ensure they are version-controlled and easily updatable alongside the code and documentation.

---

## 2. System Architecture

### 2.1. Component Architecture (C4 Model - Level 2)

**Requirement ID**: D-ARCH-01

This diagram illustrates the major components of the system, their primary responsibilities, the technology choices, and the high-level interactions between them. The architecture is based on a microservices pattern, where each service is a self-contained component deployed independently.

```mermaid
--- 
title: System Architecture - C4 Component Diagram
--- 
C4Component
  Person(engineer, "Process/Yield Engineer", "Analyzes wafer defects, provides feedback, and monitors system performance.")
  Person(data_scientist, "Data Scientist", "Manages, trains, and validates AI models. Oversees the MLOps lifecycle.")
  System_Ext(mes, "Manufacturing Execution System (MES)", "Source of wafer data via SECS/GEM or file drops. Consumes final analysis results and alerts.")

  System_Boundary(wafer_analysis_system, "Wafer Defect Analysis System") {

    Component(api_gateway, "API Gateway", "Kong", "Handles external API requests, authentication (JWT), rate limiting, and routing.")

    Component(frontend, "Web UI", "React/Vite", "Provides dashboards for results visualization, expert feedback, and system administration.")

    Component(ingestion_service, "Data Ingestion Service", "Python/FastAPI", "Parses incoming wafer map files (SECS/GEM, CSV, KLA) and initiates the processing workflow.")

    Component(preprocessing_service, "Preprocessing Service", "Python/CUDA", "Normalizes coordinates, generates wafer map images, and performs data quality checks on GPU.")

    Component(inference_service, "Inference Service", "NVIDIA Triton Server", "Hosts and executes the TensorRT-optimized AI models (YOLOv10, DeiT) for detection and classification.")

    Component(gan_service, "GAN Service", "Python/PyTorch", "Generates synthetic wafer defect maps using StyleGAN2 to augment training data.")

    Component(retraining_service, "Retraining Service", "Kubeflow/MLflow", "Orchestrates the automated model retraining, validation, and deployment pipeline.")

    ComponentDb(database, "Database", "PostgreSQL", "Stores all structured data: wafer metadata, inference results, user feedback, and model versioning information.")

    ComponentDb(storage, "Object Storage", "MinIO/S3", "Stores all binary data: raw wafer maps, processed images, synthetic images, and trained model artifacts.")

    ComponentQueue(message_queue, "Message Queue", "RabbitMQ", "Decouples services via asynchronous event-driven communication (e.g., AMQP).")
  }

  Rel(engineer, frontend, "Views dashboards, submits feedback", "HTTPS")
  Rel(data_scientist, frontend, "Manages models, triggers retraining", "HTTPS")
  Rel(frontend, api_gateway, "Makes API calls", "HTTPS/JSON")

  Rel(mes, api_gateway, "Submits wafers, retrieves results", "HTTPS/JSON")
  Rel(api_gateway, mes, "Publishes alerts", "Webhook")

  Rel(api_gateway, ingestion_service, "Routes /ingest requests")
  Rel(api_gateway, frontend, "Serves the web application")

  Rel(ingestion_service, message_queue, "Publishes [WaferReceived] event")
  Rel(ingestion_service, storage, "Saves raw wafer map file")

  Rel_Back(preprocessing_service, message_queue, "Consumes [WaferReceived] event")
  Rel(preprocessing_service, storage, "Saves processed image")
  Rel(preprocessing_service, database, "Writes wafer metadata")
  Rel(preprocessing_service, message_queue, "Publishes [ImageReadyForInference] event")

  Rel_Back(inference_service, message_queue, "Consumes [ImageReadyForInference] event")
  Rel(inference_service, storage, "Reads processed image")
  Rel(inference_service, database, "Writes inference results")
  Rel(inference_service, message_queue, "Publishes [InferenceComplete] event")

  Rel(retraining_service, database, "Reads training data & feedback")
  Rel(retraining_service, storage, "Reads images, saves new models")
  Rel(retraining_service, inference_service, "Deploys updated models to")
```

---

## 3. Data Flow Diagrams

### 3.1. Context Diagram (Level 0)

**Requirement ID**: D-DFD-01

This diagram shows the system as a single black box, illustrating its boundaries and high-level interactions with external entities.

```mermaid
--- 
title: Data Flow Diagram (Level 0 - Context)
--- 
graph TD
    subgraph External Entities
        A[Manufacturing Execution System <br/> (MES)]
        B[Process/Yield Engineer]
        C[Data Scientist]
    end

    subgraph System
        P1(Wafer Defect <br/> Analysis System)
    end

    A -- "Wafer Map Data (SECS/GEM, Files)" --> P1
    P1 -- "Analysis Results & Alerts" --> A

    B -- "Feedback & Corrections" --> P1
    P1 -- "Dashboards & Visualizations" --> B

    C -- "Model Management Commands" --> P1
    P1 -- "Training Logs & Performance Metrics" --> C
```

### 3.2. Real-Time Inference Data Flow (Level 1)

**Requirement ID**: D-DFD-02

This diagram details the flow of data between components during the real-time analysis of a single wafer.

```mermaid
--- 
title: Data Flow Diagram (Level 1 - Real-Time Inference)
--- 
graph TD
    A[External Source <br/> (MES/User)] -- "Wafer Map File" --> P1(1.0 Ingest Data)
    
    P1 -- "Raw Wafer Map" --> D1[D1: Object Storage <br/> (raw-wafer-maps)]
    P1 -- "New Wafer Event" --> P2(2.0 Preprocess Wafer)
    
    P2 -- "Raw Wafer Map" --> D1
    P2 -- "Processed Image & Metadata" --> D2[D2: Object Storage <br/> (processed-images)]
    P2 -- "Wafer Record" --> D3[D3: Database <br/> (wafers table)]
    P2 -- "Image Ready Event" --> P3(3.0 Perform Inference)
    
    P3 -- "Processed Image" --> D2
    P3 -- "AI Models" --> D4[D4: Object Storage <br/> (model-artifacts)]
    P3 -- "Inference Result" --> D3
    P3 -- "Result Ready Event" --> P4(4.0 Publish Results)
    
    P4 -- "Inference Result" --> D3
    P4 -- "Final Result & Alert" --> E1[External Sink <br/> (MES/User)]
```

### 3.3. Model Retraining Data Flow (Level 1)

**Requirement ID**: D-DFD-03

This diagram illustrates the data flow during the automated model retraining workflow, from data collection to deployment.

```mermaid
--- 
title: Data Flow Diagram (Level 1 - Model Retraining)
--- 
graph TD
    D1[D1: Database <br/> (wafers, user_feedback)] -- "New Labeled Data & Feedback" --> P1(1.0 Collect Training Data)
    D2[D2: Object Storage <br/> (processed-images)] -- "Existing Images" --> P1
    
    P1 -- "Curated Training Set" --> D3[D3: Object Storage <br/> (training-datasets)]
    P1 -- "Training Job Request" --> P2(2.0 Train & Validate Model)
    
    P2 -- "Training Set" --> D3
    P2 -- "Synthetic Data Request" --> P3(3.0 Generate Synthetic Data)
    P3 -- "Synthetic Images" --> D2
    
    P2 -- "Training Logs & Metrics" --> D4[D4: Database <br/> (mlflow_runs)]
    P2 -- "New Model Artifact" --> D5[D5: Object Storage <br/> (model-artifacts)]
    P2 -- "Validation Complete Event" --> P4(4.0 Deploy Model)
    
    P4 -- "New Model Artifact" --> D5
    P4 -- "Deployment Status" --> D1
    P4 -- "Updated Model" --> E1[Inference Service]
```

---

## 4. Sequence Diagrams

### 4.1. Real-Time Wafer Analysis Sequence

**Requirement ID**: D-SEQ-01

This diagram shows the chronological sequence of interactions between components when a new wafer is submitted for analysis.

```mermaid
--- 
title: Sequence Diagram - Real-Time Wafer Analysis
--- 
sequenceDiagram
    actor User/MES
    participant Gateway as API Gateway
    participant Ingestion as Ingestion Svc
    participant MQ as Message Queue
    participant Preproc as Preprocessing Svc
    participant Inference as Inference Svc
    participant DB as Database
    participant Storage as Object Storage

    User/MES->>Gateway: POST /api/v1/ingest (file)
    Gateway->>Ingestion: Forward request
    Ingestion->>Storage: Save raw file
    Ingestion->>MQ: publish(WaferReceived)
    Ingestion-->>Gateway: 202 Accepted (job_id)
    Gateway-->>User/MES: job_id

    MQ-->>Preproc: deliver(WaferReceived)
    activate Preproc
    Preproc->>Storage: Get raw file
    Preproc->>Preproc: Normalize & generate image (GPU)
    Preproc->>Storage: Save processed image
    Preproc->>DB: Create wafer record
    Preproc->>MQ: publish(ImageReadyForInference)
    deactivate Preproc

    MQ-->>Inference: deliver(ImageReadyForInference)
    activate Inference
    Inference->>Storage: Get processed image
    Inference->>Inference: Run model ensemble (Triton)
    Inference->>DB: Save results (patterns, confidence)
    Inference->>MQ: publish(InferenceComplete)
    deactivate Inference

    loop Poll for results
        User/MES->>Gateway: GET /api/v1/results/{job_id}
        Gateway->>DB: Query results for job_id
        DB-->>Gateway: Return results
        Gateway-->>User/MES: 200 OK (results)
    end
```

### 4.2. Expert Feedback and Active Learning Sequence

**Requirement ID**: D-SEQ-02

This diagram details the process of an engineer providing feedback on a classification, which is then used for active learning.

```mermaid
--- 
title: Sequence Diagram - Expert Feedback Loop
--- 
sequenceDiagram
    actor Engineer
    participant UI as Web UI
    participant Gateway as API Gateway
    participant DB as Database
    participant Retraining as Retraining Svc

    Engineer->>UI: Open low-confidence wafer result
    UI->>Gateway: GET /api/v1/results/{job_id}
    Gateway->>DB: Fetch result details
    DB-->>Gateway: Return result
    Gateway-->>UI: Display result & explainability map

    Engineer->>UI: Submit correction (e.g., change label)
    UI->>Gateway: POST /api/v1/feedback
    Gateway->>DB: Save feedback record (job_id, corrected_label)
    DB-->>Gateway: 201 Created
    Gateway-->>UI: Confirmation
    UI-->>Engineer: Show success message

    alt Sufficient new feedback collected
        Retraining->>DB: Query for new feedback
        DB-->>Retraining: Return new feedback records
        Retraining->>Retraining: Trigger automated retraining pipeline
    end
```

### 4.3. Automated Model Retraining and Deployment Sequence

**Requirement ID**: D-SEQ-03

This diagram illustrates the MLOps pipeline for automatically retraining, validating, and deploying an improved model.

```mermaid
--- 
title: Sequence Diagram - Automated Model Retraining & Deployment
--- 
sequenceDiagram
    participant Monitor as Drift Monitor
    participant Retraining as Retraining Svc (Kubeflow)
    participant MLflow
    participant Storage as Object Storage
    participant Triton as Triton Inference Svc

    Monitor->>Monitor: Detects 2% accuracy drop
    Monitor->>Retraining: Trigger retraining pipeline
    activate Retraining
    Retraining->>MLflow: create_run()
    Retraining->>Storage: Collect new data & feedback
    Retraining->>Retraining: Train new model version
    Retraining->>MLflow: log_metrics(accuracy, loss)
    Retraining->>MLflow: log_artifact(model.pt)
    Retraining->>Storage: Save new model artifact

    Retraining->>Triton: Deploy new model to shadow slot
    note right of Retraining: A/B testing starts. A fraction of traffic is sent to the new model.
    Retraining->>Triton: Compare performance (old vs. new)
    
    alt New model is >1% better
        Retraining->>Triton: Promote new model to production
        Retraining->>MLflow: set_tag("deployed", "true")
    else Old model is better
        Retraining->>Triton: Decommission new model
        Retraining->>MLflow: set_tag("archived", "true")
    end
    deactivate Retraining
```

---

## 5. Data Model

### 5.1. Entity-Relationship Diagram (ERD)

**Requirement ID**: D-ERD-01

This ERD outlines the schema for the PostgreSQL database, defining the core tables and their relationships.

```mermaid
--- 
title: Database Schema - Entity-Relationship Diagram
--- 
erDiagram
    users {
        INTEGER id PK
        VARCHAR(255) username
        VARCHAR(255) hashed_password
        INTEGER role_id FK
        TIMESTAMP created_at
    }

    roles {
        INTEGER id PK
        VARCHAR(50) name UNIQUE
    }

    wafers {
        INTEGER id PK
        VARCHAR(50) lot_id
        INTEGER wafer_number
        VARCHAR(100) process_step
        VARCHAR(100) equipment_id
        TIMESTAMP inspection_timestamp
        JSONB metadata
        VARCHAR(255) raw_file_path
        VARCHAR(255) processed_image_path
    }

    inference_results {
        INTEGER id PK
        INTEGER wafer_id FK
        INTEGER model_version_id FK
        VARCHAR(50) primary_pattern
        FLOAT confidence
        JSONB all_detected_patterns
        JSONB root_cause_analysis
        TIMESTAMP created_at
    }

    user_feedback {
        INTEGER id PK
        INTEGER result_id FK
        INTEGER user_id FK
        VARCHAR(50) corrected_pattern
        TEXT notes
        TIMESTAMP created_at
    }

    model_versions {
        INTEGER id PK
        VARCHAR(20) version_string
        VARCHAR(255) model_path
        JSONB hyperparameters
        JSONB performance_metrics
        INTEGER dataset_id FK
        TIMESTAMP created_at
    }

    training_datasets {
        INTEGER id PK
        VARCHAR(20) version_string
        INTEGER real_sample_count
        INTEGER synthetic_sample_count
        JSONB class_distribution
        TIMESTAMP created_at
    }

    users ||--o{ roles : "has"
    wafers ||--|{ inference_results : "has"
    inference_results ||--o{ user_feedback : "receives"
    user_feedback }|--|| users : "is given by"
    model_versions ||--|{ inference_results : "produces"
    training_datasets ||--o{ model_versions : "is used by"
```

---

## 6. Deployment Architecture

### 6.1. Kubernetes Deployment Diagram

**Requirement ID**: D-DEPLOY-01

This diagram illustrates how the microservices are deployed as pods within a Kubernetes cluster, distributed across CPU and GPU nodes.

```mermaid
--- 
title: Kubernetes Deployment Architecture
--- 
graph TD
    subgraph Internet/Fab Network
        direction LR
        A[User/MES]
    end

    subgraph Kubernetes Cluster
        B(Ingress Controller) --> C{API Gateway Service}

        subgraph CPU Node 1
            D1[Pod: API Gateway]
            D2[Pod: Web UI]
            D3[Pod: Ingestion Svc]
        end

        subgraph CPU Node 2
            E1[Pod: Retraining Svc]
            E2[Pod: RabbitMQ]
            E3[Pod: PostgreSQL]
        end

        subgraph GPU Node 1 (NVIDIA H100)
            F1[Pod: Preprocessing Svc]
            F2[Pod: Inference Svc (Triton)]
        end

        subgraph GPU Node 2 (NVIDIA H100)
            G1[Pod: Preprocessing Svc]
            G2[Pod: Inference Svc (Triton)]
        end

        C --> D1
        C --> D2
        C --> D3

        E2 -- "Persistent Volume" --> H1(PV/StorageClass)
        E3 -- "Persistent Volume" --> H1

        F1 -- "GPU Access" --> I1(NVIDIA Device Plugin)
        F2 -- "GPU Access" --> I1
        G1 -- "GPU Access" --> I1
        G2 -- "GPU Access" --> I1
    end

    subgraph Cloud/On-Prem Services
        J[Object Storage (S3/MinIO)]
    end

    A --> B
    D3 -.-> J
    F1 -.-> J
    G1 -.-> J
    F2 -.-> J
    G2 -.-> J
```

This detailed design specification provides a complete blueprint for the development team, covering all critical aspects of the system from high-level architecture to low-level data models and deployment strategies.
