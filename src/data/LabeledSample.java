package data;

public class LabeledSample {
	public double[] x;
	public double y;
	public double weight;
	public boolean isSampled; // 样本采样是否采到
	public boolean isSplitToLeft; // 是否划分到左半边的树
}
