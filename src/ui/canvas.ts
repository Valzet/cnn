import * as tf from "@tensorflow/tfjs";

export type DigitImageData = {
  imageData: ImageData;
  digit: number;
};

export function preprocessImage(imageData: ImageData): tf.Tensor4D {
  const pixels = tf.browser.fromPixels(imageData, 1);

  const resized = tf.image.resizeBilinear(pixels, [28, 28]);

  const normalized = resized.div(255.0);

  const batched = normalized.expandDims(0);

  return batched as tf.Tensor4D;
}

export function displayDigitOnCanvas(
  imageData: ImageData, 
  digit: number, 
  digitCanvasCtx: CanvasRenderingContext2D | null
) {
  if (!digitCanvasCtx) {
    console.error("Canvas context not initialized");
    return;
  }

  // Scale the 28x28 image to fit the canvas (140x140)
  const scale = 5;
  const scaledCanvas = document.createElement("canvas");
  const scaledCtx = scaledCanvas.getContext("2d")!;

  scaledCanvas.width = 28 * scale;
  scaledCanvas.height = 28 * scale;

  scaledCtx.putImageData(imageData, 0, 0);

  digitCanvasCtx.imageSmoothingEnabled = false;
  digitCanvasCtx.clearRect(
    0,
    0,
    digitCanvasCtx.canvas.width,
    digitCanvasCtx.canvas.height,
  );
  digitCanvasCtx.drawImage(
    scaledCanvas,
    0,
    0,
    28,
    28,
    0,
    0,
    28 * scale,
    28 * scale,
  );

  const digitLabelElement = document.getElementById("digit-label");
  if (digitLabelElement) {
    digitLabelElement.textContent = `Цифра: ${digit}`;
  }
}

export function generateRandomDigitImage(): DigitImageData {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;

  canvas.width = 28;
  canvas.height = 28;

  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const digit = Math.floor(Math.random() * 10); // NUM_CLASSES = 10

  ctx.font = "bold 20px Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "white";
  ctx.fillText(digit.toString(), canvas.width / 2, canvas.height / 2);

  return {
    imageData: ctx.getImageData(0, 0, canvas.width, canvas.height),
    digit: digit,
  };
}