import * as tf from "@tensorflow/tfjs";
import {
  INPUT_SHAPE,
  KERNEL_SIZE,
  FILTERS_LAYER_1,
  FILTERS_LAYER_2,
  POOL_SIZE,
  NUM_CLASSES
} from "../modelConfig";

export function createModel(): tf.LayersModel {
  const model = tf.sequential({
    layers: [
      tf.layers.conv2d({
        inputShape: INPUT_SHAPE,
        kernelSize: KERNEL_SIZE,
        filters: FILTERS_LAYER_1,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling",
      }),

      tf.layers.maxPooling2d({
        poolSize: POOL_SIZE,
        strides: POOL_SIZE,
      }),

      tf.layers.conv2d({
        kernelSize: KERNEL_SIZE,
        filters: FILTERS_LAYER_2,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling",
      }),

      tf.layers.maxPooling2d({
        poolSize: POOL_SIZE,
        strides: POOL_SIZE,
      }),

      tf.layers.flatten(),

      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: "varianceScaling",
        activation: "softmax",
      }),
    ],
  });

  return model;
}

export function visualizeLayers(model: tf.LayersModel) {
  const layerInfo: { name: string; className: string; config: any }[] = [];
  model.layers.forEach((layer) => {
    layerInfo.push({
      name: layer.name,
      className: layer.getClassName(),
      config: layer.getConfig(),
    });
  });

  console.log("Информация о слоях:", layerInfo);
}