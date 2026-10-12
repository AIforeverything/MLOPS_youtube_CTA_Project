const form = document.getElementById("predictionForm");

const button = document.getElementById("predictButton");
const buttonText = document.getElementById("buttonText");
const spinner = document.getElementById("spinner");

const resultCard = document.getElementById("resultCard");
const predictionValue = document.getElementById("predictionValue");
const errorMessage = document.getElementById("errorMessage");


// Display error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
}


// Clear error message
function clearError() {
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
}


// Handle loading state
function setLoading(loading) {
    button.disabled = loading;

    spinner.classList.toggle("hidden", !loading);

    buttonText.textContent = loading
        ? "Predicting..."
        : "Predict View Count";
}


// Format prediction using Indian number formatting
function formatNumber(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return String(value);
    }

    return new Intl.NumberFormat("en-IN", {
        maximumFractionDigits: 0
    }).format(number);
}


// Handle prediction form submission
form.addEventListener("submit", async function (event) {

    // Prevent page reload
    event.preventDefault();

    clearError();
    resultCard.classList.add("hidden");

    // Collect input features
    const payload = {
        category: document.getElementById("category").value,

        subscriber_count: Number(
            document.getElementById("subscriber_count").value
        ),

        channel_view_count: Number(
            document.getElementById("channel_view_count").value
        ),

        duration_seconds: Number(
            document.getElementById("duration_seconds").value
        )
    };


    // Validate category
    if (!payload.category) {
        showError("Please select a category.");
        return;
    }


    // Validate numerical inputs
    const numericFields = [
        "subscriber_count",
        "channel_view_count",
        "duration_seconds"
    ];

    for (const field of numericFields) {
        if (
            !Number.isFinite(payload[field]) ||
            payload[field] < 0
        ) {
            showError(
                `Please enter a valid non-negative value for ${
                    field.replaceAll("_", " ")
                }.`
            );
            return;
        }
    }


    setLoading(true);


    try {

        // Send JSON to Flask
        const response = await fetch("/predict", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(payload)
        });


        // Flask returns JSON
        const result = await response.json();


        // Handle Flask errors
        if (!response.ok) {
            throw new Error(
                result.error ||
                `Prediction failed (HTTP ${response.status}).`
            );
        }


        // Validate prediction
        if (
            typeof result.prediction !== "number" ||
            !Number.isFinite(result.prediction)
        ) {
            throw new Error(
                "The server did not return a valid prediction."
            );
        }


        // Display predicted view count
        predictionValue.textContent = formatNumber(
            result.prediction
        );

        resultCard.classList.remove("hidden");


    } catch (error) {

        console.error("Prediction error:", error);

        showError(
            error.message ||
            "Unable to get prediction. Please check the Flask server."
        );

    } finally {

        setLoading(false);
    }

});