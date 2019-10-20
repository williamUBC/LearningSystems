package ece.cpen502;
import org.junit.Assert;
import org.junit.Test;
public class NeuralNetTester {
    @Test
    public void testSigmoid(){
        int [] x = {0,1,-1,2,-2,10,-10};
        double [] expectedOutput = {0,0.462117157,-0.462117157,0.761594156,-0.761594156,0.999909204,-0.999909204};
        int testInputSize = x.length;
        double [] actualOutput = new double[testInputSize];
        NeuralNet testNN = new NeuralNet(2,4,1);
        for(int i=0;i<testInputSize;i++){
            actualOutput[i] = testNN.sigmoid(x[i]);
        }
        Assert.assertArrayEquals(expectedOutput,actualOutput,0.001);
    }

    @Test
    public void testForwardfeed(){
        NeuralNet testNN = new NeuralNet(2,4,1);
        double input[] = {1,1};
        testNN.zeroWeights();
        double actualOutput[] = testNN.forwardFeed(input);
        double expectedOutput[] = {0.7089};
        Assert.assertEquals(expectedOutput[0],actualOutput[0],0.01);
    }
}
