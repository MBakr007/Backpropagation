/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package backpropagationneuralnetworks;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;
import java.util.Vector;

/**
 *
 * @author mbakr
 */

public class BackpropagationNeuralNetworks 
{

    public static double M/*number of Input Nodes*/ , L /*number of Hidden Nodes*/ , N/* number of Output Nodes*/;
    public static int numberOfTrainingExamples = 0;
    public static double leanringRate = 0.3;
    public static Vector<double[]> trainingExamples = new Vector<>();
    public static Vector<double[]> testingExamples = new Vector<>();
    public static Vector<double[]> hiddenNodesWeight = new Vector<>();
    public static Vector<double[]> outputWeight = new Vector<>();
    public static double[][] hiddenNodes;
    
    public static void initializeHiddenNode()
    {
        hiddenNodes = new double[(int)L][1];
    }
    public static double getRandomNumber_double(double min, double max) 
    {
        Random r = new Random();
        double random = min + r.nextFloat()* (max - min);
        return random;
    }
    
    public static void initializeWeightHiddenNodes()
    {
        for(int i = 0; i < M; i++)
        {
            double[] arr = new double[(int)L];
            for(int j = 0; j < L; j++)
            {
                arr[j] = getRandomNumber_double(1, 5);
            }
            hiddenNodesWeight.add(arr);
        }
    }
    public static void initializeWeightOutput()
    {
        for(int i = 0; i < L; i++)
        {
            double[] arr = new double[(int)N];
            for(int j = 0; j < N; j++)
            {
                arr[j] = getRandomNumber_double(1, 5);
            }
            outputWeight.add(arr);
        }
    }
    
    public static double[] convert(String str, int size)
    {
        String s = "";
        double[] numbers = new double[size];
        int index = 0;
        for(int i = 0; i < str.length(); i++)
        {
            if(str.charAt(i) != ' ')
                s+= str.charAt(i);
            else if(s.length() > 0)
            {
                numbers[index] = Float.valueOf(s);
                index++;
                s = "";
            }
        }
        if(s.length() > 0)
            numbers[index] = Double.valueOf(s);
        return numbers;
    }
    public static double calculateSD(double numArray[], int size)
    {
        double standardDeviation = 0;
        int length = size;
        

        double mean = caculateMean(numArray, size);

        for(int i = 0; i < size; i++) {
            standardDeviation += Math.pow(numArray[i] - mean, 2);
        }

        return  Math.sqrt(standardDeviation/length);
    }
    public static float caculateMean(double[] arr, int size)
    {
        float sum = 0;
        for(int i = 0; i < size; i++)
            sum += arr[i];
        
        return sum/size;
        
    }
    public static Vector<double[]> normalization(Vector<double[]> vec, int size1, int size2)
    {
        for(int i = 0; i < size1; i++)
        {
            double[] arr = new double[size2];
            for(int j = 0; j < size2; j++)
            {
                arr[j] = vec.get(j)[i];
            }
            
            double SD = calculateSD(arr, size2);
            double mean = caculateMean(arr, size2);
            
            for(int j = 0; j < size2; j++)
                arr[j] = (arr[j] - mean) / SD;
            
            for(int j = 0; j < size2; j++)
            {
                vec.get(j)[i] = arr[j];
            }
        }
        return vec;
    }
    public static void readTrainingExamples(String fileName) throws FileNotFoundException, IOException
    {
        File file = new File(fileName);
        BufferedReader br = new BufferedReader(new FileReader(file));
        String st;
        int i = 0;
        while ((st = br.readLine()) != null) 
        {
            if(i == 0)
            {
                double[] arr = convert(st, 3);
                M = arr[0];
                L = arr[1];
                N = arr[2];
                i++;
            }
            else if(i == 1)
            {
                numberOfTrainingExamples = Integer.valueOf(st);
                i++;
            }
            else
            {
                double[] arr = convert(st, (int) (M+N));
                trainingExamples.add(arr);
                
            }
        }
    }
    public static double[][] multiplyMatrix(double[][] mat1, double[][] mat2, int row1, int col1, int row2, int col2)
    {
        double result[][] = new double[row1][col2]; 
  
        // Multiply the two marices 
        for (int i = 0; i < row1; i++) 
        { 
            for (int j = 0; j < col2; j++) 
            { 
                for (int k = 0; k < row2; k++) 
                    result[i][j] += mat1[i][k] * mat2[k][j]; 
            } 
        }
        return result;
    }
    public static double[][] activationFunction(double[][] arr, int row, int col)
    {
        // sigmoid function
        double[][] result = new double[row][col];
        for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
            {
                result[i][j] = (1/(1+Math.exp(-1*arr[i][j])));
            }
        }
        return result;
    }
    public static double[][] feedForwad(double[] input)
    {
        double[][] weightMatrix = new double[(int)L][(int)M];
        for(int i = 0; i < L; i++)
        {
            for(int j = 0; j < M; j++)
            {
                weightMatrix[i][j] = hiddenNodesWeight.get(j)[i];
            }
        }
        double[][] inputMatrix = new double[(int)M][1];
        for(int i = 0; i < M; i++)
        {
            inputMatrix[i][0] = input[i];
        }
        
        hiddenNodes = multiplyMatrix(weightMatrix, inputMatrix, (int)L, (int)M, (int)M, 1);
        
        hiddenNodes = activationFunction(hiddenNodes, (int)L, 1);
        
        double[][] weightOutputMatrix = new double[(int)N][(int)L];
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < L; j++)
            {
                weightOutputMatrix[i][j] = outputWeight.get(j)[i];
            }
        }
        double[][] outputMatrix = new double[(int)N][1];
        outputMatrix = multiplyMatrix(weightOutputMatrix, hiddenNodes,(int)N, (int)L, (int)L, 1);
        outputMatrix = activationFunction(outputMatrix, (int)N, 1);
        
        return outputMatrix;
    }
    public static void backPropagation(double[][] outputMatrix, double[] trainingExample) 
    {
        double[] deltaOutput = new double[(int)N];
        for(int i = 0; i < N; i++)
        {
            deltaOutput[i] = (outputMatrix[i][0] - trainingExample[(int)(M+i)]) * outputMatrix[i][0] * (1 - outputMatrix[i][0]);
        }
        
        double[] deltaHidden = new double[(int)L];
        for(int i = 0; i < L; i++)
        {
            double sum = 0;
            for(int j = 0; j < N; j++)
            {
                sum += deltaOutput[j] * outputWeight.get(i)[j];
            }
            deltaHidden[i] = sum * hiddenNodes[i][0] * (1 - hiddenNodes[i][0]);
        }
        // update output weights 
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < L; j++)
            {
                outputWeight.get(j)[i] = outputWeight.get(j)[i] - leanringRate * deltaOutput[i] * hiddenNodes[j][0];
            }
        }
        // update hidden weights
        for(int i = 0; i < L; i++)
        {
            for(int j = 0; j < M; j++)
            {
                hiddenNodesWeight.get(j)[i] = hiddenNodesWeight.get(j)[i] - leanringRate * deltaHidden[i] *  trainingExample[j];
            }
        }
    }
    public static void clearFile(String fileName) throws IOException
    {
        BufferedWriter writer = Files.newBufferedWriter(Paths.get(fileName));
	writer.write("");
	writer.flush();
    }
    public static void appendStrToFile(String fileName, String str) 
    { 
        try 
        { 
            // Open given file in append mode. 	
            BufferedWriter out = new BufferedWriter(new FileWriter(fileName, true));
            out.write(str); 
            out.close(); 
        } 
        catch (IOException e) { 
            System.out.println("exception occoured" + e); 
        } 
    }
    public static void writeWeightsToFile()
    {
        for(int i = 0; i < hiddenNodesWeight.size(); i++)
        {
            String str = "";
            for(int j = 0; j < hiddenNodesWeight.get(i).length; j++)
            {
                str += hiddenNodesWeight.get(i)[j] + " ";
            }
            str += "\n";
            appendStrToFile("hidden nodes weights.txt", str);
        }
        for(int i = 0; i < outputWeight.size(); i++)
        {
            String str = "";
            for(int j = 0; j < outputWeight.get(i).length; j++)
            {
                str += outputWeight.get(i)[j] + " ";
            }
            str += "\n";
            appendStrToFile("output weights.txt", str);
        }
        
    }
    public static void readWeightsFromFile(String fileName, String str) throws FileNotFoundException, IOException
    {
        File file = new File(fileName);
        BufferedReader br = new BufferedReader(new FileReader(file));
        String st;
        int i = 0;
        while ((st = br.readLine()) != null) 
        {
            if(str.equals("hidden"))
                hiddenNodesWeight.set(i,convert(st, (int)L));
            else
                outputWeight.set(i, convert(st, (int)N));
            i++;
        }
    }
    public static void testing(String fileName, int size) throws IOException
    {
        readWeightsFromFile("hidden nodes weights.txt", "hidden");
        readWeightsFromFile("output weights.txt", "output");
        File file = new File(fileName);
        BufferedReader br = new BufferedReader(new FileReader(file));
        String st;
        int i = 0;
        while ((st = br.readLine()) != null) 
        {
            testingExamples.set(i, convert(st, (int)(M+N)));
        }
        testingExamples = normalization(testingExamples, (int)(M+N), size);
        
    }
    public static void main(String[] args) throws IOException  
    {
        readTrainingExamples("train.txt");
        trainingExamples = normalization(trainingExamples,(int)(M+N), numberOfTrainingExamples);
        initializeHiddenNode();
        initializeWeightHiddenNodes();
        initializeWeightOutput();
        
//        boolean bool = true;
        for(int j = 0; j < numberOfTrainingExamples; j++)
        {
            for(int i = 0; i < 500; i++) 
            {
                double[][] arr = feedForwad(trainingExamples.get(j));
//                if(bool)
//                {
//                    for(int a = 0; a < N; a++)
//                        System.out.println(arr[a][0]);
//                    bool = false;
//                }
                backPropagation(arr, trainingExamples.get(j));
            }
        }
        clearFile("hidden nodes weights.txt");
        clearFile("output weights.txt");
        writeWeightsToFile();
        
        // Testing
        int size = 515;
        testing("test.txt", size);
        
        double[] arr = new double[numberOfTrainingExamples];
        double[][] predicted = new double[numberOfTrainingExamples][1];
        Vector<double[][]> vec = new Vector<>();
        for(int i = 0; i < numberOfTrainingExamples; i++)
        {
            predicted = feedForwad(testingExamples.get(i));
            vec.add(predicted);
        }
        for(int i = 0; i < numberOfTrainingExamples; i++)
        {
            double sum = 0;
            System.out.println("Training Example No." + (i + 1));
            for(int j = 0; j < N; j++)
            {
                double num = testingExamples.get(i)[j] - vec.get(i)[j][0];
                sum += Math.pow(num, 2);
            }
            System.out.println("MSE = " + sum / N);
            System.out.println("Acutal Value  = " + testingExamples.get(i)[(int)(M+N-1)]);
            System.out.println("Predicted Vaule = " + vec.get(i)[0][0]);
            System.out.println("//////////////////////////////////////////////");
        }
    }
}


