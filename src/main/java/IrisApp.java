import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.eclipse.collections.impl.bag.sorted.mutable.SynchronizedSortedBag;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.LossFunction;
import org.nd4j.linalg.api.ops.impl.controlflow.While;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import play.libs.F;

import javax.xml.crypto.Data;
import java.io.File;
import java.io.IOException;

public class IrisApp {

    public static void main(String[] args) throws IOException, InterruptedException {
        double leaninRate=0.001;    //pourcentage d'erreur
        int numInputs=4;      //nbr de critère (kaykono f dakhla u homa li kaymayzo bin fleur et autre)
        int numHidden=4;    //nbr de neurons ou couches(li kaykono wasst)
        int numOutputs=3;    //nbr de sorties (3 iris 'fleurs')
        System.out.println("Création du modèle");
        //Créer Configuration de notre modèle
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .updater(new Adam(leaninRate))
                .list()
                .layer(0, new DenseLayer.Builder()
                            .nIn(numInputs)
                            .nOut(numHidden)
                            .activation(Activation.SIGMOID)
                            .build()
                )
                .layer(0, new OutputLayer.Builder()
                        .nIn(numHidden)
                        .nOut(numOutputs)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                        .build()
                )
                .build();
        //Créer le modèle et le lier par la config
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        //Afficher la config
            //System.out.println(configuration.toJson());

        /* Démarrage du serveur de monitoring du processus d'apprentissage
            Ce serveur web permet de consulter une page web avec des graphiques
            qui montre l'évolution de l'entrainement de modèle
         */
    /*    UIServer uiServer = UIServer.getInstance();
        InMemoryStatsStorage inMemoryStatsStorage = new InMemoryStatsStorage();
        uiServer.attach(inMemoryStatsStorage);
        model.setListeners(new StatsListener(inMemoryStatsStorage));    */

        System.out.println("Entrainement du modèle");

        File fileTrain = new ClassPathResource("iris-train.csv").getFile();
        RecordReader recordReaderTrain = new CSVRecordReader();
        recordReaderTrain.initialize(new FileSplit(fileTrain));
        int batchSize = 1;  // 1 input puis son output
        int classIndex = 4;
        DataSetIterator dataSetIteratorTrain =
                new RecordReaderDataSetIterator(recordReaderTrain, batchSize, classIndex, numOutputs);

        //Afficher le contenu du dataset
     /*   while (dataSetIteratorTrain.hasNext()){
            System.out.println("-------------------------");
            DataSet dataSet = dataSetIteratorTrain.next();
            System.out.println(dataSet.getFeatures());
            System.out.println(dataSet.getLabels());
        }   */

        //lancer entrainement du modèle
        int nbrEpocs = 100;  // nbr de cycle d'apprentissage (cad 3adad marat li kay3awd y apprenez notre model les données hta ywali hafdhom)
        for (int i = 0; i < nbrEpocs; i++) {
            model.fit(dataSetIteratorTrain);
        }




        System.out.println("Evaluation du modèle");

        File fileTest = new ClassPathResource("iris-test.csv").getFile();
        RecordReader recordReaderTest = new CSVRecordReader();
        recordReaderTest.initialize(new FileSplit(fileTest));
        DataSetIterator dataSetIteratorTest =
                new RecordReaderDataSetIterator(recordReaderTest, batchSize, classIndex, numOutputs);
        Evaluation evaluation = new Evaluation();
        while(dataSetIteratorTest.hasNext()){
            DataSet dataSetTest = dataSetIteratorTest.next();
            INDArray features = dataSetTest.getFeatures();
            INDArray targetLabels = dataSetTest.getLabels();
            INDArray predictedLabels = model.output(features);
            evaluation.eval(predictedLabels, targetLabels);
        }
        System.out.println(evaluation.stats());

        //Une fois que le modèle est appris, on doit le enregistrer sous-forme zip
        ModelSerializer.writeModel(model, "irisModel.zip", true);




    }
}
