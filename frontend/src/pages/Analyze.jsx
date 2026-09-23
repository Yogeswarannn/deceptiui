import { useState } from "react";
import { useNavigate } from "react-router-dom";

function Analyze() {
  const [image, setImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const navigate = useNavigate();

  const handleFile = (file) => {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      alert("Please upload an image file.");
      return;
    }

    const preview = URL.createObjectURL(file);

    setImage({
      file: file,
      preview: preview,
    });
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    handleFile(file);
  };

  const handleDrop = (event) => {
    event.preventDefault();

    const file = event.dataTransfer.files[0];
    handleFile(file);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleAnalyze = async () => {
    if (!image || !image.file) return;

    setIsAnalyzing(true);

    try {
      const formData = new FormData();
      formData.append("image", image.file);

      const response = await fetch("http://localhost:5000/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`);
      }

      const data = await response.json();

      navigate("/results", {
        state: {
          image: image.preview,
          analysisData: data,
        },
      });
    } catch (error) {
      console.error("Error analyzing image:", error);
      alert("An error occurred during analysis. Make sure the backend server is running.");
      setIsAnalyzing(false);
    }
  };

  return (
    <main className="analyze-page">

      <div className="analyze-container">

        <div className="page-heading">
          <p className="eyebrow">
            SCREENSHOT ANALYSIS
          </p>

          <h1>Analyze an Interface</h1>

          <p>
            Upload a screenshot and DeceptiUI will analyze it
            for potential dark patterns.
          </p>
        </div>

        {isAnalyzing ? (

          <div className="analysis-loading">

            <div className="loading-spinner"></div>

            <h2>Analyzing Interface...</h2>

            <p>
              Detecting visual and textual evidence.
            </p>

          </div>

        ) : (

          <div className="upload-section">

            {!image ? (

              <div
                className="upload-box"
                onDrop={handleDrop}
                onDragOver={handleDragOver}
              >

                <div className="upload-icon">
                  ↑
                </div>

                <h2>Upload Screenshot</h2>

                <p>
                  Drag and drop an image here
                  <br />
                  or
                </p>

                <label className="upload-button">
                  Choose Image

                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    hidden
                  />
                </label>

                <span className="upload-hint">
                  PNG, JPG or JPEG
                </span>

              </div>

            ) : (

              <div className="preview-section">

                <h2>Screenshot Preview</h2>

                <div className="preview-box">

                  <img
                    src={image.preview}
                    alt="Uploaded screenshot"
                  />

                </div>

                <div className="preview-actions">

                  <label className="secondary-button">
                    Choose Another

                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileChange}
                      hidden
                    />
                  </label>

                  <button
                    className="primary-button"
                    onClick={handleAnalyze}
                  >
                    Analyze Screenshot
                    <span>→</span>
                  </button>

                </div>

              </div>

            )}

          </div>

        )}

      </div>

    </main>
  );
}

export default Analyze;