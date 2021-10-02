import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

public class IrisPrediction {
    public static void main(String[] args) throws IOException {

        //Charger le modèle
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("irisModel.zip"));


        System.out.println("Prédiction");

        //Les données qu'on veut prédicter
        INDArray inputData = Nd4j.create(new double[][]{
                {5.1, 3.5, 1.4, 0.2},
                {6.7, 3.1, 4.4, 1.4},
                {6.0, 3.0, 4.8, 1.8}
        });
        INDArray output = model.output(inputData);
        System.out.println(output);   //  [ 0.9862, 0.0137, 0.0038]
        //Pour organiser affichage = classe : nom du iris
        int[] classes = output.argMax(1).toIntVector();
        String[] labels = {"Iris-setosa", "Iris-versicplor", "Iris-virginica"};
        for (int i = 0; i < classes.length ; i++) {
            System.out.println("Classe :"+labels[classes[i]]);
        }
    }
}
