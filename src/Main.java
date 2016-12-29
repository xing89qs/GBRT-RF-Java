import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Scanner;

import boosting.GradientBoostingRegressor;
import rf.RandomForestRegressor;

public class Main {
	public static void main(String[] args) throws Exception {
		HashMap<String, Object> params = new HashMap<>();
		params.put("learning_rate", 0.05);
		params.put("n_estimator", 100);
		params.put("max_depth", 3);
		params.put("objective", "ls");
		params.put("feature_rate", 0.1);
		params.put("sample_rate", 0.8);
		params.put("spliter", "leaf");
		params.put("num_leafs", 8);
		// GradientBoostingRegressor gbdt = new
		// GradientBoostingRegressor(params);
		RandomForestRegressor gbdt = new RandomForestRegressor(params);

		double trainX[][] = new double[26964][155];
		double trainY[] = new double[26964];
		double trainWeight[] = new double[26964];
		Arrays.fill(trainWeight, 1.0);
		Scanner in = new Scanner(new BufferedReader(new FileReader(new File("/Users/mac/Desktop/train_frame1.csv"))));
		in.nextLine();
		int numLines = 0;
		while (in.hasNextLine()) {
			String line = in.nextLine();
			String[] str = line.split(",", -1);
			int featureIndex = 0;
			for (int i = 3; i < str.length; i++) {
				if (i == 138)
					continue;
				if (str[i].equals("") || str[i] == null)
					trainX[numLines][featureIndex++] = -1000;
				else
					trainX[numLines][featureIndex++] = Double.valueOf(str[i]);
			}
			trainY[numLines] = Double.valueOf(str[2]);
			numLines++;
			assert (featureIndex == 155);
		}
		System.out.println(numLines);
		in.close();

		double testX[][] = new double[4464][155];
		double testY[] = new double[4464];
		in = new Scanner(new BufferedReader(new FileReader(new File("/Users/mac/Desktop/test_frame1.csv"))));
		in.nextLine();
		numLines = 0;
		while (in.hasNextLine()) {
			String line = in.nextLine();
			String[] str = line.split(",", -1);
			int featureIndex = 0;
			for (int i = 3; i < str.length; i++) {
				if (i == 138)
					continue;
				if (str[i].equals("") || str[i] == null)
					testX[numLines][featureIndex++] = -1000;
				else
					testX[numLines][featureIndex++] = Double.valueOf(str[i]);
			}
			assert (featureIndex == 155);
			testY[numLines] = Double.valueOf(str[2]);
			numLines++;
		}
		System.out.println(numLines);
		in.close();
		gbdt.fit(trainX, trainY, trainWeight);
		double[][] ans = gbdt.predict(testX);
		double[] ret = new double[ans[0].length];
		for (int i = 0; i < testX.length; i++) {
			ret[i] = 0;
			for (int j = 0; j < ans.length; j++) {
				ret[i] += ans[j][i];
			}
			ret[i] /= ans.length;
			//System.out.println(ret[i] + " " + testY[i]);
		}

		// --------------------------------------------
		// gbdt.fit(trainX, trainY, trainWeight,null);
		// double[] ret = gbdt.predict(testX);

		double all = 0;
		for (int i = 0; i < testX.length; i++) {
			double error = testY[i] - ret[i];
			all += error * error;
		}
		System.out.println(all);
	}
}
