# import the necessary packages
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2

conf = tf.ConfigProto()

MODEL_PATH = '/home/dhivakar/work/projects/main/cotton_detection/tf_obj_det/research/object_detection/saved_models/P_051/frozen_inference_graph_P051_tf1-15-3.pb'
LAYER_NAME = 'FeatureExtractor/InceptionV2/InceptionV2/Mixed_5b/concat'
IMAGE_PATH = '/home/dhivakar/work/projects/main/data/br_v2.1/may03/cd_night_imgs/set1/frame-501.jpg'


class GradCAM:
        def __init__(self, model_path, layerName=None):
                # store the model, the class index used to measure the class
                # activation map, and the layer to be used when visualizing
                # the class activation map
                # self.model = model
                self.graph = self.load_graph(model_path)
                self.layerName = layerName

                # if the layer name is None, attempt to automatically find
                # the target output layer
                # if self.layerName is None:
                        # self.layerName = self.find_target_layer()

        def load_graph(self, pb_file_path):
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(pb_file_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
            return detection_graph

        def find_target_layer(self):
                # attempt to find the final convolutional layer in the network
                # by looping over the layers of the network in reverse order
                for layer in reversed(self.model.layers):
                        # check to see if the layer has a 4D output
                        if len(layer.output_shape) == 4:
                                return layer.name

                # otherwise, we could not find a 4D layer so the GradCAM
                # algorithm cannot be applied
                raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

        def compute_heatmap(self, image, eps=1e-8):
                # Use graph and session to get the conv outputs
                with tf.GradientTape() as tape:
                    with self.graph.as_default():
                        with tf.Session(config=conf) as sess:
                            ops = tf.get_default_graph().get_operations()
                            all_tensor_names = {output.name for op in ops for output in op.outputs}
                            tensor_dict = {}
                            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                                tensor_name = key + ':0'
                                if tensor_name in all_tensor_names:
                                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                            tensor_dict['mixed5b'] = tf.get_default_graph().get_tensor_by_name(self.layerName + ':0')

                            output_dict = sess.run(tensor_dict,
                                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

                            output_dict['num_detections'] = int(output_dict['num_detections'][0])
                            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                            output_dict['detection_scores'] = output_dict['detection_scores'][0]
                            convOutputs = output_dict['mixed5b']

                            # use automatic differentiation to compute the gradients
                            grad_dict = {
                                'detection_boxes': tf.convert_to_tensor(output_dict['detection_boxes']),
                                'detection_scores': tf.convert_to_tensor(output_dict['detection_scores'])
                            }
                            convOutputs = tf.convert_to_tensor(convOutputs)
                            grads = tape.gradient(grad_dict, convOutputs)

                # construct our gradient model by supplying (1) the inputs
                # to our pre-trained model, (2) the output of the (presumably)
                # final 4D layer in the network, and (3) the output of the
                # softmax activations from the model
                # gradModel = Model(
                        # inputs=[self.model.inputs],
                        # outputs=[self.model.get_layer(self.layerName).output,
                                # self.model.output])

                #record operations for automatic differentiation
                # with tf.GradientTape() as tape:
                        # # cast the image tensor to a float-32 data type, pass the
                        # # image through the gradient model, and grab the loss
                        # # associated with the specific class index
                        # inputs = tf.cast(image, tf.float32)
                        # (convOutputs, predictions) = gradModel(inputs)
                        # loss = predictions[:, self.classIdx]

                # compute the guided gradients
                castConvOutputs = tf.cast(convOutputs > 0, "float32")
                castGrads = tf.cast(grads > 0, "float32")
                guidedGrads = castConvOutputs * castGrads * grads

                # the convolution and guided gradients have a batch dimension
                # (which we don't need) so let's grab the volume itself and
                # discard the batch
                convOutputs = convOutputs[0]
                guidedGrads = guidedGrads[0]

                # compute the average of the gradient values, and using them
                # as weights, compute the ponderation of the filters with
                # respect to the weights
                weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
                cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

                # grab the spatial dimensions of the input image and resize
                # the output class activation map to match the input image
                # dimensions
                (w, h) = (image.shape[2], image.shape[1])
                heatmap = cv2.resize(cam.numpy(), (w, h))

                # normalize the heatmap such that all values lie in the range
                # [0, 1], scale the resulting values to the range [0, 255],
                # and then convert to an unsigned 8-bit integer
                numer = heatmap - np.min(heatmap)
                denom = (heatmap.max() - heatmap.min()) + eps
                heatmap = numer / denom
                heatmap = (heatmap * 255).astype("uint8")

                # return the resulting heatmap to the calling function
                return heatmap

        def overlay_heatmap(self, heatmap, image, alpha=0.5,
                colormap=cv2.COLORMAP_VIRIDIS):
                # apply the supplied color map to the heatmap and then
                # overlay the heatmap on the input image
                heatmap = cv2.applyColorMap(heatmap, colormap)
                output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
                # return a 2-tuple of the color mapped heatmap and the output,
                # overlaid image
                return (heatmap, output)


if __name__ == '__main__':
    image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gradcam = GradCAM(model_path=MODEL_PATH, layerName=LAYER_NAME)
    gradcam.compute_heatmap(image)
