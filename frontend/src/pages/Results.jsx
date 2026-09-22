import { Link, useLocation } from "react-router-dom";

function Results() {
  const location = useLocation();

  const uploadedImage = location.state?.image;

  const predictions = [
    {
      label: "Default Choice",
      confidence: 0.91,
      description:
        "The interface appears to steer the user toward a pre-selected option.",
    },
    {
      label: "False Hierarchy",
      confidence: 0.84,
      description:
        "One option appears to receive stronger visual emphasis than the alternatives.",
    },
    {
      label: "Privacy Zuckering",
      confidence: 0.78,
      description:
        "The interface may encourage the user to share more information than necessary.",
    },
  ];

  return (
    <main className="results-page">

      <div className="results-container">

        <div className="results-header">

          <div>

            <p className="eyebrow">
              ANALYSIS COMPLETE
            </p>

            <h1>Detection Results</h1>

            <p>
              DeceptiUI identified potential dark patterns
              in the uploaded interface.
            </p>

          </div>

          <div className="result-status">
            <span className="status-dot"></span>
            Analysis Complete
          </div>

        </div>


        <div className="results-summary">

          <div>
            <span className="summary-number">
              {predictions.length}
            </span>

            <span className="summary-label">
              Potential Patterns
            </span>
          </div>

          <div>
            <span className="summary-number">
              91%
            </span>

            <span className="summary-label">
              Highest Confidence
            </span>
          </div>

        </div>


        <section className="visual-analysis">

          <div className="section-title">

            <h2>Visual Analysis</h2>

            <p>
              Screenshot and visual evidence used by the model.
            </p>

          </div>

          <div className="visual-grid">

            <div className="screenshot-panel">

              <div className="panel-header">
                <span>UPLOADED SCREENSHOT</span>
              </div>

              <div className="screenshot-container">

                {uploadedImage ? (

                  <img
                    src={uploadedImage}
                    alt="Uploaded interface"
                  />

                ) : (

                  <div className="no-image">
                    No screenshot available
                  </div>

                )}

              </div>

            </div>


            <div className="evidence-panel">

              <div className="panel-header">
                <span>VISUAL EVIDENCE</span>
              </div>

              <div className="visual-evidence-content">

                <div className="evidence-placeholder">
                  Visual evidence regions will appear here.
                </div>

                <p>
                  The model can highlight interface regions
                  that contributed to each prediction.
                </p>

              </div>

            </div>

          </div>

        </section>


        <section className="predictions-section">

          <div className="section-title">

            <h2>Detected Patterns</h2>

            <p>
              Predictions are shown with their confidence scores.
            </p>

          </div>

          <div className="prediction-grid">

            {predictions.map((prediction) => (

              <div
                className="prediction-card"
                key={prediction.label}
              >

                <div className="prediction-top">

                  <h3>
                    {prediction.label}
                  </h3>

                  <span className="confidence">
                    {Math.round(
                      prediction.confidence * 100
                    )}%
                  </span>

                </div>

                <div className="confidence-bar">

                  <div
                    className="confidence-fill"
                    style={{
                      width: `${prediction.confidence * 100}%`,
                    }}
                  ></div>

                </div>

                <p>
                  {prediction.description}
                </p>

              </div>

            ))}

          </div>

        </section>


        <section className="evidence-section">

          <div className="evidence-panel">

            <div className="panel-header">
              <span>TEXTUAL EVIDENCE</span>
            </div>

            <div className="evidence-quote">
              "Continue with recommended settings"
            </div>

            <p>
              This text contributed to the Default Choice
              prediction.
            </p>

          </div>


          <div className="evidence-panel">

            <div className="panel-header">
              <span>WHY WAS THIS DETECTED?</span>
            </div>

            <p className="explanation">
              The interface appears to steer the user toward
              a pre-selected option. Visual emphasis and
              interface wording both contribute to the prediction.
            </p>

          </div>

        </section>


        <div className="results-actions">

          <Link
            to="/analyze"
            className="primary-button"
          >
            Analyze Another Screenshot
            <span>→</span>
          </Link>

        </div>

      </div>

    </main>
  );
}

export default Results;