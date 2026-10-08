```javascript
const form = document.getElementById("predictionForm");

const button = document.getElementById("predictButton");
const buttonText = document.getElementById("buttonText");
const spinner = document.getElementById("spinner");

const resultCard = document.getElementById("resultCard");
const predictionValue = document.getElementById("predictionValue");
const errorMessage = document.getElementById("errorMessage");


function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
}


function clearError() {
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
}


function setLoading(loading) {
    button.disabled = loading;

    spinner.classList.toggle("hidden", !loading);

    buttonText.textContent = loading
        ? "Predicting..."
        : "Predict View Count";
}


function formatNumber(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return String(value);
    }

    return new Intl.NumberFormat("en-IN", {
        maximumFractionDigits: 0
    }).format(number);
}


form.addEventListener("submit", async function (event) {

    // Prevent normal HTML form submission
    event.preventDefault();

    clearError();
    resultCard.classList.add("hidden");

    // Collect input values
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
    if (
        !Number.isFinite(payload.subscriber_count) ||
        !Number.isFinite(payload.channel_view_count) ||
        !Number.isFinite(payload.duration_seconds)
    ) {
        showError("Please enter valid numeric values.");
        return;
    }


    setLoading(true);


    try {

        // Send data to Flask
        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(payload)
        });


        if (!response.ok) {

            const message = await response.text();

            throw new Error(
                message || `Prediction failed (${response.status}).`
            );
        }


        /*
         * Your current Flask /predict route returns
         * index.html rather than JSON.
         *
         * Therefore we receive the returned HTML
         * and extract #predictionValue from it.
         */
        const html = await response.text();


        const parsedHTML =
            new DOMParser().parseFromString(html, "text/html");


        const returnedPrediction =
            parsedHTML.querySelector("#predictionValue");


        if (!returnedPrediction) {
            throw new Error(
                "Prediction was returned, but the result could not be read."
            );
        }


        // Display prediction
        predictionValue.textContent =
            formatNumber(returnedPrediction.textContent.trim());


        resultCard.classList.remove("hidden");


    } catch (error) {

        console.error("Prediction error:", error);

        showError(
            error.message || "Unable to get prediction."
        );

    } finally {

        setLoading(false);
    }

});
```