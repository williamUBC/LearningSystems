package ece.cpen502;

import java.io.File;
import java.io.IOException;

public class NeuralNet implements NeuralNetInterface{

    private int argNumInputs;
    private int argNumHidden;
    private int argNumOutput;
    private double argLearningRate = 0.2;
    private double argMomentumTerm = 0.0;
    private double argA;
    private double argB;
    private double [][] weightsInputToHidden;
    private double [][] weightsHiddenToOutput;
    static double bias = 1;
    private double[] hiddenValues;
    private double[] outputValue;
//    private double[][] inputValue = {
//            {0, 0},
//            {0, 1},
//            {1, 0},
//            {1, 1}
//    };
    //constructor
    public NeuralNet(int argNumInputs, int argNumHidden, int argNumOutput){
        this.argNumInputs = argNumInputs;
        this.argNumHidden = argNumHidden;
        this.argNumOutput = argNumOutput;
        this.weightsInputToHidden = new double[argNumInputs + 1][argNumHidden];
        this.weightsHiddenToOutput = new double[argNumHidden + 1][argNumOutput];
        this.hiddenValues = new double[argNumHidden];
        this.outputValue = new double[argNumOutput];
    }

    @Override
    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x)) ;
    }

    @Override
    public double customSigmoid(double x) {
        return (argB - argA)/(1 + Math.exp(-x)) + argA;
    }

    @Override
    public void initializeWeights() {
        for(int i = 0; i<argNumInputs+1; i++){
            for (int j=0; j<argNumHidden; j++){
                this.weightsInputToHidden[i][j] = Math.random() - 0.5;
                //System.out.println(this.weightsInputToHidden[i][j]);
            }
        }

        for(int i = 0; i<argNumHidden+1; i++){
            for (int j=0; j<argNumOutput; j++){
                this.weightsHiddenToOutput[i][j] = Math.random() - 0.5;
                //System.out.println(this.weightsHiddenToOutput[i][j]);
            }
        }
    }

    @Override
    public void zeroWeights() {
        for(int i = 0; i<this.argNumInputs+1; i++){
            for (int j=0; j<this.argNumHidden; j++){
                this.weightsInputToHidden[i][j] = 0.5;
                //System.out.println(this.weightsInputToHidden[i][j]);
            }
        }

        for(int i = 0; i<this.argNumHidden+1; i++){
            for (int j=0; j<this.argNumOutput; j++){
                this.weightsHiddenToOutput[i][j] = 0.5;
                //System.out.println(this.weightsHiddenToOutput[i][j]);
            }
        }
    }

    public double[] forwardFeed(double[] input){
        //hiddenValues[] = new double[argNumHidden];
        //double outputValue[] = new double[argNumOutput];
        //this.inputValue = input;
        if(input.length != argNumInputs){
            throw new ArrayIndexOutOfBoundsException();
        }else{
            //calculate value for nodes in hidden layer
            for (int hiddenNodeIndex=1; hiddenNodeIndex<argNumHidden+1; hiddenNodeIndex++){
                hiddenValues[hiddenNodeIndex-1] = 0.0;
                for(int inputNodeIndex=1; inputNodeIndex<argNumInputs+1; inputNodeIndex++){
                    hiddenValues[hiddenNodeIndex-1] += input[inputNodeIndex-1] * weightsInputToHidden[inputNodeIndex][hiddenNodeIndex-1];
                }
                hiddenValues[hiddenNodeIndex-1] += weightsInputToHidden[0][hiddenNodeIndex-1];
                hiddenValues[hiddenNodeIndex-1] = sigmoid(hiddenValues[hiddenNodeIndex-1]);
            }
            //calculate value for nodes in output layer
            for(int outputNodeIndex=0; outputNodeIndex<argNumOutput; outputNodeIndex++){
                outputValue[outputNodeIndex] = 0.0;
                for (int hiddenNodeIndex=1; hiddenNodeIndex<argNumHidden+1; hiddenNodeIndex++){
                    outputValue[outputNodeIndex] += hiddenValues[hiddenNodeIndex-1] * weightsHiddenToOutput[hiddenNodeIndex][outputNodeIndex];
                }
                outputValue[outputNodeIndex] += weightsHiddenToOutput[0][outputNodeIndex];
                outputValue[outputNodeIndex] = sigmoid(outputValue[outputNodeIndex]);
            }
        }
        return outputValue;
    }



    public void backPropagation(double[] expectedOut, double[] actualOut, double[] inputValue){
        //double delta_OutputToHidden[] = new double[argNumOutput];
        double delta_Hidden[] = new double[argNumHidden];
        double delta_Output[] = new double[argNumOutput];
        double delta_Output_beforeSigmoid[] = new double[argNumOutput];
        //double updateAmountIH[][] = new double[argNumInputs][argNumHidden];
        //double updateAmountHO[][] = new double[argNumHidden][argNumOutput];
        if(actualOut.length!=argNumOutput && expectedOut.length!=argNumOutput){
            throw new ArrayIndexOutOfBoundsException();
        }else{
            //get delta at output nodes

            for(int j=0; j<argNumOutput; j++){
                //delta_Output_beforeSigmoid[j] = 0.5 * Math.pow(expectedOut[j] - actualOut[j],2);
                delta_Output_beforeSigmoid[j] = expectedOut[j] - actualOut[j];
                delta_Output[j] = actualOut[j] * (1 - actualOut[j]) * delta_Output_beforeSigmoid[j];
            }

            //get delta at hidden nodes
            for(int i=0; i<argNumHidden; i++){
                delta_Hidden[i] = 0.0;
                for(int j=0; j<argNumOutput; j++){
                    delta_Hidden[i] += hiddenValues[i] * (1 - hiddenValues[i]) * delta_Output_beforeSigmoid[j] * weightsHiddenToOutput[i+1][j];
                    //System.out.println(actualOut[j] * (1 - actualOut[j]));
                }
            }

            //update weights
            //calculate value for nodes in hidden layer
            for (int hiddenNodeIndex=0; hiddenNodeIndex<argNumHidden; hiddenNodeIndex++){
                for(int inputNodeIndex=1; inputNodeIndex<argNumInputs+1; inputNodeIndex++){

                    weightsInputToHidden[inputNodeIndex][hiddenNodeIndex] += argLearningRate * delta_Hidden[hiddenNodeIndex] * inputValue[inputNodeIndex-1];
                }
                weightsInputToHidden[0][hiddenNodeIndex] += argLearningRate * delta_Hidden[hiddenNodeIndex];
                //System.out.println(weightsInputToHidden[0][hiddenNodeIndex]);
            }

            //calculate value for nodes in output layer

            for(int outputNodeIndex=0; outputNodeIndex<argNumOutput; outputNodeIndex++){
                for (int hiddenNodeIndex=1; hiddenNodeIndex<argNumHidden+1; hiddenNodeIndex++){
                    weightsHiddenToOutput[hiddenNodeIndex][outputNodeIndex] += argLearningRate * delta_Output[outputNodeIndex] * hiddenValues[hiddenNodeIndex-1];
                }
                weightsHiddenToOutput[0][outputNodeIndex] += argLearningRate * delta_Output[outputNodeIndex];
                //System.out.println(weightsHiddenToOutput[0][outputNodeIndex]);
            }
            //momentum
        }


    }
    @Override
    public double outputFor(double[] X) {
        return 0;
    }

    @Override
    public double train(double[] X, double argValue) {
        return 0;
    }

    @Override
    public void save(File argFile) {

    }

    @Override
    public void load(String argFileName) throws IOException {

    }

    public static void main(String[] args) {
        NeuralNet myNN = new NeuralNet(2, 4, 1);
        double x_train[][] = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };
        double y_label[][] = {
                {0},
                {1},
                {1},
                {0}
        };
        //int epoch = 0;
        myNN.initializeWeights();
        double[] res = new double[0];
        double myError = 0.0;
        double epoch = 0.0;
        //System.out.println(x_train[1][1]);
        //myNN.zeroWeights();
        //for (int epoch = 0; epoch < 50000; epoch++) {
         do{
            for (int i = 0; i < 4; i++) {
                res = myNN.forwardFeed(x_train[i]);
                myError = 0.0;
                //System.out.println(y_label[i][0]);
                //System.out.println(x_train[i][1]);
                myNN.backPropagation(y_label[i], res, x_train[i]);
                myError += 0.5 * Math.pow((res[0] - y_label[i][0]), 2);
                //System.out.println(res[0]);

            }
            epoch += 1;
            //System.out.println(res[0]);
            //System.out.println("Epoch: " + epoch + "    Error: " + myError);
             System.out.println(myError);
        }while (myError>0.01 && epoch<500000);
    }
}

