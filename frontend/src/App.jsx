import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Analyze from "./pages/Analyze";
import Results from "./pages/Results";
import Navbar from "./components/Navbar";

function Home() {
  return (
    <main className="home">

      {/* ================= HERO ================= */}

      <section className="hero cinematic-hero" id="home">

        <div className="hero-glow glow-one"></div>
        <div className="hero-glow glow-two"></div>

        <div className="hero-content">

          <div className="eyebrow">
            <span className="status-dot"></span>
            EXPLAINABLE AI FOR USER INTERFACES
          </div>

          <h1>
            Detect Dark
            <br />
            <span>Patterns.</span>
          </h1>

          <h2>Understand Why.</h2>

          <p className="hero-description">
            DeceptiUI analyzes interfaces through visual and textual
            evidence to uncover potentially deceptive design patterns
            and explain what contributed to each detection.
          </p>

          <div className="hero-actions">

            <Link
              to="/analyze"
              className="primary-button"
            >
              Analyze Screenshot
              <span>→</span>
            </Link>

          </div>

          <div className="hero-meta">

            <span>VISION</span>
            <span>OCR</span>
            <span>MULTIMODAL AI</span>
            <span>EXPLAINABILITY</span>

          </div>

        </div>


        {/* HERO VISUAL */}

        <div className="hero-visual">

          <div className="visual-orbit orbit-one"></div>
          <div className="visual-orbit orbit-two"></div>

          <div className="interface-card">

            <div className="interface-topbar">
              <span></span>
              <span></span>
              <span></span>
            </div>

            <div className="interface-content">

              <div className="interface-label">
                INTERFACE ANALYSIS
              </div>

              <div className="fake-heading">
                Choose your settings
              </div>

              <div className="fake-line"></div>
              <div className="fake-line short"></div>

              <div className="fake-options">

                <div className="fake-option selected">
                  <div className="fake-radio"></div>

                  <div>
                    <strong>Recommended</strong>
                    <small>Best experience</small>
                  </div>
                </div>

                <div className="fake-option">
                  <div className="fake-radio"></div>

                  <div>
                    <strong>Customize</strong>
                    <small>Choose manually</small>
                  </div>
                </div>

              </div>

              <div className="fake-button">
                Continue
              </div>

            </div>

            <div className="detection-tag">
              <span></span>
              DEFAULT CHOICE
              <strong>91%</strong>
            </div>

          </div>


          <div className="floating-label label-one">
            <span>01</span>
            VISUAL EVIDENCE
          </div>

          <div className="floating-label label-two">
            <span>02</span>
            TEXTUAL EVIDENCE
          </div>

        </div>

      </section>


      {/* ================= HOW IT WORKS ================= */}

      <section
        className="story-section"
        id="how-it-works"
      >

        <div className="section-heading">

          <div>
            <p className="eyebrow">
              THE PROCESS
            </p>

            <h2>
              From screenshot
              <br />
              <span>to explanation.</span>
            </h2>
          </div>

          <p>
            DeceptiUI combines computer vision, OCR and multimodal
            analysis to move beyond simple classification.
          </p>

        </div>


        <div className="process-grid">

          <div className="process-card">

            <span className="process-number">01</span>

            <div className="process-icon">↑</div>

            <h3>Upload</h3>

            <p>
              Provide a screenshot of the interface you want
              DeceptiUI to investigate.
            </p>

          </div>


          <div className="process-card featured-process">

            <span className="process-number">02</span>

            <div className="process-icon">◉</div>

            <h3>Analyze</h3>

            <p>
              Visual features and interface text are analyzed
              together to identify potential dark patterns.
            </p>

          </div>


          <div className="process-card">

            <span className="process-number">03</span>

            <div className="process-icon">✦</div>

            <h3>Explain</h3>

            <p>
              Results are presented with confidence scores,
              evidence and human-readable explanations.
            </p>

          </div>

        </div>

      </section>


      {/* ================= DETECTION CATEGORIES ================= */}

      <section
        className="patterns-section"
        id="patterns"
      >

        <div className="section-heading centered">

          <p className="eyebrow">
            WHAT WE DETECT
          </p>

          <h2>
            Five patterns.
            <br />
            <span>One interface.</span>
          </h2>

          <p>
            DeceptiUI currently focuses on five categories of
            potentially deceptive interface behaviour.
          </p>

        </div>


        <div className="pattern-showcase">

          <div className="pattern-card pattern-large">

            <span>01</span>

            <h3>Hard to Close</h3>

            <p>
              Interfaces that make dismissing or exiting an
              experience unnecessarily difficult.
            </p>

            <div className="pattern-line"></div>

          </div>


          <div className="pattern-card">

            <span>02</span>

            <h3>Default Choice</h3>

            <p>
              Pre-selected options that influence user decisions.
            </p>

          </div>


          <div className="pattern-card">

            <span>03</span>

            <h3>False Hierarchy</h3>

            <p>
              Visual emphasis that makes one option appear
              more important than alternatives.
            </p>

          </div>


          <div className="pattern-card">

            <span>04</span>

            <h3>Nagging</h3>

            <p>
              Repeated prompts or interruptions that push
              users toward an action.
            </p>

          </div>


          <div className="pattern-card">

            <span>05</span>

            <h3>Privacy Zuckering</h3>

            <p>
              Design that encourages users to reveal more
              information than they may intend.
            </p>

          </div>

        </div>

      </section>


      {/* ================= RESEARCH ================= */}

      <section
        className="research-section"
        id="research"
      >

        <div className="research-visual">

          <div className="research-ring ring-one"></div>
          <div className="research-ring ring-two"></div>

          <div className="research-core">
            <span>AI</span>
            <small>EXPLAIN</small>
          </div>

        </div>


        <div className="research-content">

          <p className="eyebrow">
            RESEARCH
          </p>

          <h2>
            Not just
            <br />
            <span>what. Why.</span>
          </h2>

          <p>
            A prediction alone doesn't tell us enough. DeceptiUI
            is designed to connect model outputs with visual and
            textual evidence so that users can understand the
            reasoning behind a detection.
          </p>


          <div className="research-points">

            <div>
              <span>01</span>
              <strong>Visual Evidence</strong>
              <p>
                Identify interface regions contributing to
                model predictions.
              </p>
            </div>

            <div>
              <span>02</span>
              <strong>Textual Evidence</strong>
              <p>
                Extract interface text through OCR and use it
                as supporting evidence.
              </p>
            </div>

            <div>
              <span>03</span>
              <strong>Human Explanation</strong>
              <p>
                Translate model predictions into understandable
                explanations.
              </p>
            </div>

          </div>

        </div>

      </section>


      {/* ================= FINAL CTA ================= */}

      <section className="final-section">

        <div className="final-glow"></div>

        <p className="eyebrow">
          SEE WHAT YOUR INTERFACE IS SAYING
        </p>

        <h2>
          Don't just detect.
          <br />
          <span>Understand.</span>
        </h2>

        <p>
          Upload an interface and explore the evidence behind
          its potential dark patterns.
        </p>

        <Link
          to="/analyze"
          className="primary-button"
        >
          Analyze a Screenshot
          <span>→</span>
        </Link>

      </section>


      {/* ================= FOOTER ================= */}

      <footer className="footer">

        <div className="footer-logo">
          DeceptiUI
        </div>

        <p>
          Explainable AI for User Interfaces
        </p>

        <span>
          © 2026 DeceptiUI
        </span>

      </footer>

    </main>
  );
}


function App() {
  return (
    <BrowserRouter>

      <Navbar />

      <Routes>

        <Route
          path="/"
          element={<Home />}
        />

        <Route
          path="/analyze"
          element={<Analyze />}
        />

        <Route
          path="/results"
          element={<Results />}
        />

      </Routes>

    </BrowserRouter>
  );
}

export default App;