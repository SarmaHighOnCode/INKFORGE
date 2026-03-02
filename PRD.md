# Product Requirements Document (PRD): INKFORGE — Personal Handwriting Synthesis Engine

## 1. Executive Summary

INKFORGE is a web-based, AI-driven handwriting synthesis application designed to generate long-form, full-page handwritten documents that perfectly replicate a specific user's handwriting style. Unlike existing "handwriting font" generators that produce unnatural, identical characters or single-line AI demos that lack document-level coherence, INKFORGE utilizes PyTorch LSTM models to learn the exact stroke dynamics, pen pressure, and natural variability of a user's writing. It can synthesize thousands of words with realistic margin wrapping, paragraph indentation, and simulated writing fatigue, indistinguishable from a human writer. 

The core value proposition of this product iteration is the ability to securely train the AI locally on a user's machine (e.g., an RTX 4060) to learn their distinct handwriting profile, and subsequently deploy the lightweight trained model for fast, cloud-hosted text-to-handwriting generation.

---

## 2. Target Audience & Use Cases

*   **Students/Academics:** Generating handwritten assignments or notes automatically from typed text to save time while maintaining a personal touch.
*   **D2C E-commerce Businesses:** Creating personalized "handwritten" thank-you notes at scale for packaging inserts.
*   **Marketing/Sales Professionals:** Generating personalized direct mail (e.g., real estate outreach) to dramatically increase open and response rates.
*   **Individuals:** Sending digitized, personalized letters, invitations, or greeting cards.

---

## 3. Product Features & Requirements

### 3.1 Data Collection & Onboarding
*   **Requirement:** Users must be able to easily provide the system with their handwriting profile.
*   **Solution:** A browser-based **Handwriting Data Collector Tool** (`data_collector.html`).
*   **Key Features:**
    *   Responsive HTML5 Canvas for stroke capture via mouse or tablet stylus.
    *   Pre-defined prompt sentences covering the full alphabet and numbers.
    *   Automatic generation of XML stroke data and corresponding transcription text files.
    *   Writer ID tagging to manage multiple distinct handwritings (e.g., the user vs. a friend).

### 3.2 AI Training Pipeline (Local Environment)
*   **Requirement:** The system must process the collected handwriting data and train an LSTM sequence generation model efficiently on consumer hardware (specifically targeting an RTX 4060 GPU and 16GB RAM).
*   **Key Features:**
    *   **Preprocessing Script:** Parses XML stroke data into relative `(Δx, Δy, p1, p2, p3)` coordinates required by the MDN (Mixture Density Network).
    *   **PyTorch Training Loop:** Fast, local fine-tuning or scratch training to generate a lightweight `.pt` checkpoint file representing the user's "brain" or handwriting profile.

### 3.3 Text-to-Handwriting Generation Engine (Inference)
*   **Requirement:** The core engine must take raw string input and convert it into a sequence of dynamic handwriting strokes simulating human behavior.
*   **Key Features:**
    *   **State Passing Architecture:** The LSTM hidden state is maintained across character chunks to ensure consistency across thousands of words.
    *   **Natural Variation Injectors:**
        *   *Fatigue Simulation:* Degradation in stroke precision and looseness over long text passages.
        *   *Baseline Drift:* A slow-moving sinusoidal algorithm that naturally curves lines across the page, avoiding perfectly rigid margins.
        *   *Character Variability:* The same letter looks slightly different each time based on random distribution sampling via MDN.

### 3.4 Page Layout & Rendering Engine
*   **Requirement:** The generated strokes must be mapped to a strict visual document structure and rendered into a high-quality export format.
*   **Key Features:**
    *   **Typewriter Layout Algorithm:** Deterministic math handling auto-line wrapping, paragraph indentation (e.g., 1.5cm indent), and inter-line spacing.
    *   **Multi-Page Support:** Ability to map up to 5,000+ words across continuous A4/Letter size structures.
    *   **Export Pipeline:** Renders final mathematical stroke paths to PNG (300 DPI) or PDF for realistic printing.

### 3.5 Cloud Deployment & API
*   **Requirement:** The finalized, specific handwriting model must be hostable on a cloud platform (e.g., Hugging Face Spaces) to be accessible remotely.
*   **Key Features:**
    *   **FastAPI Backend:** Orchestrates the text ingestion, task queuing (Celery/Redis), and stroke generation asynchronously.
    *   **Model Loading:** Capability to load the pre-trained `.pt` file generated from the local RTX 4060 training phase on CPU-tier cloud environments.
    *   **Webhook/WebSocket Integration:** Streaming stroke generation data to the frontend for a "live writing" user experience.

---

## 4. Technical Specifications & Architecture

### Stack
*   **ML Engine:** PyTorch 2.x, TorchScript
*   **Backend API:** Python 3.10+, FastAPI, Uvicorn
*   **Task Management:** Celery, Redis
*   **Frontend UI:** React 18, Vite 5, Zustand (State), Tailwind CSS
*   **Rendering:** CairoSVG, Pillow
*   **Infrastructure:** Docker, Docker Compose

### Hardware & Performance Parameters
*   **Training Specs (Local):** 16GB System RAM, NVIDIA RTX 4060 (8GB VRAM). Expected training time: <1 hour depending on dataset size.
*   **Inference Latency Target:** < 3 seconds per 100 characters on standard cloud CPU.
*   **Model Weights Size:** ~50MB - 150MB per writer profile (`.pt` file).

---

## 5. Security & Privacy

*   **Local Training Advantage:** Because data collection and initial ML training are performed locally on the user's GPU, highly personal and sensitive biometric handwriting data is never uploaded to a third-party server during the learning phase.
*   **Cloud Data Management:** Generated text payloads passed to the cloud API must be ephemeral, not stored persistently unless specifically requested by the user, ensuring content privacy.

---

## 6. Success Metrics (KPIs)

*   **Realism Rate:** > 80% of human testers cannot distinguish the INKFORGE output from the user's actual handwriting in a blind test.
*   **Generation Reliability:** 100% success rate in generating long-form documents (e.g., 2,000 words) without layout crashes or memory overflows.
*   **Data Collection Conversion:** Time taken for a user to successfully collect 50 training sentences using `data_collector.html` is under 15 minutes.
