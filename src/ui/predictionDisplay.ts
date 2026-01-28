import { NUM_CLASSES } from "../modelConfig";

export type Prediction = {
  digit: number;
  predictedDigit: number;
  probabilities: number[];
  timestamp: Date;
};

export let predictionHistory: Array<Prediction> = [];

export function displayPredictionResults(
  probabilities: number[],
  resultContainer: HTMLDivElement,
  autoClassification: boolean = false,
) {
  if (autoClassification) {
    resultContainer.innerHTML =
      "<h3>Результаты автоматической классификации:</h3>";
  } else {
    resultContainer.innerHTML = "<h3>Результаты классификации:</h3>";
  }

  const probabilityList = probabilities.map((prob, index) => ({
    digit: index,
    probability: prob,
  }));

  probabilityList.sort((a, b) => b.probability - a.probability);

  const topResults = probabilityList.slice(0, 5);

  const resultList = document.createElement("ul");
  topResults.forEach((item) => {
    const listItem = document.createElement("li");
    listItem.textContent = `Цифра ${item.digit}: ${(item.probability * 100).toFixed(2)}%`;
    resultList.appendChild(listItem);
  });

  resultContainer.appendChild(resultList);

  const digitLabelElement = document.getElementById("digit-label");
  let recognizedDigit = -1;
  if (digitLabelElement) {
    const text = digitLabelElement.textContent || "";
    const match = text.match(/Цифра: (\d)/);
    if (match) {
      recognizedDigit = parseInt(match[1]);
    }
  }

  const predictedDigit = probabilityList[0].digit;

  predictionHistory.push({
    digit: recognizedDigit,
    predictedDigit: predictedDigit,
    probabilities: [...probabilities],
    timestamp: new Date(),
  });

  if (predictionHistory.length > 40) {
    predictionHistory = predictionHistory.slice(-40);
  }

  const predContainer = document.getElementById("prediction-results-container");
  if (predContainer) {
    const recentPredictions = predictionHistory.slice(-10);
    predContainer.innerHTML = `
      <h3>Последние предсказания:</h3>
      <div class="recent-predictions">
        ${recentPredictions
          .map(
            (prediction) => `
          <div class="prediction-item">
            <div class="prediction-header">
              <span class="timestamp">${prediction.timestamp.toLocaleTimeString()}</span>
              <span class="digit-pair">Цифра: ${prediction.digit} → Предсказано: ${prediction.predictedDigit}</span>
            </div>
            <div class="prediction-match">${prediction.digit === prediction.predictedDigit ? "✓ Верно" : "✗ Ошибка"}</div>
          </div>
        `,
          )
          .join("")}
      </div>
    `;
  }

  updateProbabilityBars(probabilities);
}

export function updateProbabilityBars(probabilities: number[]) {
  const probabilityBarsContainer = document.getElementById("probability-bars");
  if (!probabilityBarsContainer) return;

  probabilityBarsContainer.innerHTML = "";

  for (let i = 0; i < NUM_CLASSES; i++) {
    const probability = probabilities[i] || 0;
    const percentage = (probability * 100).toFixed(2);

    const barContainer = document.createElement("div");
    barContainer.className = "probability-bar-container";

    barContainer.innerHTML = `
      <div class="probability-label">${i}</div>
      <div class="probability-bar">
        <div class="probability-fill" style="width: ${percentage}%"></div>
        <span class="probability-percent">${percentage}%</span>
      </div>
    `;

    probabilityBarsContainer.appendChild(barContainer);
  }
}