import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class IndexedLinkedListEntropy {
    
    private static final int NUMBER_OF_RUNS = 100;
    // 1681184501458 -> r = -0.9291
    private static final long SEED = 1681184501458L;
    
    public static void main(String[] args) {
        Fingers fingers = new Fingers(new Random(SEED));
        double bestCorrelation = Double.NEGATIVE_INFINITY;
        double worstCorrelation = Double.POSITIVE_INFINITY;
        
        int bestCorrelationIteration = -1;
        int worstCorrelationIteration = -1;
        
        double averageCorrelation = 0.0;
        double[] correlations = new double[100];
        
        for (int i = 1; i <= 100; i++) {
            System.out.println("# Iteration: " + i);
            List<DataPoint> dataPoints = simulate(fingers);
            Collections.sort(dataPoints);
            
            for (DataPoint dp : dataPoints) {
                final String str = String.format("%.3f %.3f\n",
                                                 dp.getEntropy(),
                                                 dp.getAmortizedWork()).replaceAll(",", ".");
                System.out.print(str);
            }
            
            final double correlation = 
                    pearsonCorrelationCoefficient(dataPoints);
            
            correlations[i - 1] = correlation;
            averageCorrelation += correlation;
            
            if (bestCorrelation < correlation) {
                bestCorrelation = correlation;
                bestCorrelationIteration = i;
            }
            
            if (worstCorrelation > correlation) {
                worstCorrelation = correlation;
                worstCorrelationIteration = i;
            }
            
            System.out.printf("# r = %.4f\n\n", correlation);
            
            fingers.reset();
        }
        
        System.out.printf("Best correlation: %f, i = %d.\n",
                          bestCorrelation, 
                          bestCorrelationIteration);
        
        System.out.printf("Worst correlation: %f, i = %d.\n",
                          worstCorrelation, 
                          worstCorrelationIteration);
        
        double avg = averageCorrelation / 100.0;
        double std = computeStdSum(avg, correlations);
       
        System.out.printf("Average correlation: %.4f\n", avg);
        System.out.printf("Correlation Std:     %.4f\n", std);
    }
    
    private static double computeStdSum(double avg, double[] correlations) {
        double sum = 0.0;
        
        for (double c : correlations) {
            sum += (c - avg) * (c - avg);
        }
        
        return Math.sqrt(sum / correlations.length);
    }
    
    private static List<DataPoint> simulate(Fingers fingers) {
        List<DataPoint> dataPoints = 
                new ArrayList<>(NUMBER_OF_RUNS);
        
        for (int i = 0; i < NUMBER_OF_RUNS; i++) {
            DataPoint result = fingers.simulate();
            dataPoints.add(result); 
        }
        
        return dataPoints;
    }
    
    private static double pearsonCorrelationCoefficient
        (List<DataPoint> dataPoints) {

        double averageX = computeMeanX(dataPoints);
        double averageY = computeMeanY(dataPoints);

        double numerator = computeNumerator(dataPoints, averageX, averageY);
        double denominator = computeDenominator(dataPoints, averageX, averageY);

        return numerator / denominator;
    }

    private static double 
        computeNumerator(List<DataPoint> dataPoints,
                         double averageX, 
                         double averageY) {
        double sum = 0.0;

        for (DataPoint dataPoint : dataPoints) {
            sum += (dataPoint.getEntropy()       - averageX)
                 * (dataPoint.getAmortizedWork() - averageY);
        }

        return sum;
    }

    private static double 
        computeDenominator(List<DataPoint> dataPoints,
                           double averageX, 
                           double averageY) {
        double factor1Sum = 0.0;
        double factor2Sum = 0.0;

        for (DataPoint dataPoint : dataPoints) {
            double dx = dataPoint.getEntropy()       - averageX;
            double dy = dataPoint.getAmortizedWork() - averageY;
            factor1Sum += dx * dx;
            factor2Sum += dy * dy;
        }

        return Math.sqrt(factor1Sum * factor2Sum);
    }

    private static double computeMeanX(List<DataPoint> dataPoints) {
        double sum = 0.0;

        for (DataPoint dataPoint : dataPoints) {
            sum += dataPoint.getEntropy();
        }

        return sum / dataPoints.size();
    }

    private static double computeMeanY(List<DataPoint> dataPoints) {
        double sum = 0.0;

        for (DataPoint dataPoint : dataPoints) {
            sum += dataPoint.getAmortizedWork();
        }

        return sum / dataPoints.size();
    }
}

class Fingers {
    
    static final int NUMBER_OF_FINGERS = 100;
    
    private static final int LIST_SIZE = NUMBER_OF_FINGERS * NUMBER_OF_FINGERS;
    
    private final int[] fingerIndices = new int[NUMBER_OF_FINGERS];
    private final Random random;
    private int b = 0;
    
    Fingers(Random random) {
        this.random = random;
    }
    
    void reset() {
        b = 0;
        
        for (int i = 0; i < fingerIndices.length; i++) {
            fingerIndices[i] = i;
        }
    }
    
    DataPoint simulate() {
        prepareFingersArray();
        return process();
    }
    
    private void prepareFingersArray() {
        if (++b > NUMBER_OF_FINGERS) {
            throw new IllegalStateException("Search exhausted.");
        }
        
        int a = random.nextInt(LIST_SIZE - b * NUMBER_OF_FINGERS + 1);
        
        for (int i = 0; i < fingerIndices.length; i++) {
            fingerIndices[i] = a + b * i;
        }
    }
    
    private DataPoint process() {
        int totalWork = 0;
        
        for (int targetIndex = 0; targetIndex < LIST_SIZE; targetIndex++) {
            int closestFingerIndex = findClosestFingerIndex(targetIndex);
            totalWork += Math.abs(closestFingerIndex - targetIndex);
        }
        
        return new DataPoint(computeEntropy(), 1.0 * totalWork / LIST_SIZE);
    }
    
    private int findClosestFingerIndex(int targetIndex) {
        int closestDistance = Integer.MAX_VALUE;
        int closestFingerIndex = -1;
        
        for (int fingerIndex : fingerIndices) {
            int distance = Math.abs(targetIndex - fingerIndex);
            
            if (closestDistance > distance) {
                closestDistance = distance;
                closestFingerIndex = fingerIndex;
            }
        }
        
        return closestFingerIndex;
    }
    
    private double computeEntropy() {
        double squareRootOfListSize = Math.sqrt(LIST_SIZE);
        double sums = 0.0;
        
        for (int i = 0; i < NUMBER_OF_FINGERS - 1; i++) {
            sums += Math.abs(fingerIndices[i + 1] - fingerIndices[i] 
                                                  - squareRootOfListSize);
        }
        
        double tmp = sums / LIST_SIZE;
        
        if (tmp > 1.0) {
            System.out.println("tmp: " + tmp);
        }
        
        return 1.0 - sums / LIST_SIZE;
    }
}

class DataPoint implements Comparable<DataPoint> {
    private final double entropy;
    private final double amortizedWork;
    
    DataPoint(double entropy, double amortizedWork) {
        this.entropy = entropy;
        this.amortizedWork = amortizedWork;
    }
    
    @Override
    public String toString() {
        return String.format("%.05f %d", entropy, amortizedWork)
                     .replace(',', '.');
    }
    
    double getEntropy() {
        return entropy;
    }
    
    double getAmortizedWork() {
        return amortizedWork;
    }

    @Override
    public int compareTo(DataPoint o) {
        return Double.compare(entropy, o.entropy);
    }
}
