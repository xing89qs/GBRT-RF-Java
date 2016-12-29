package util.dump;

// 基于对应的数据表

public class TreeInfo {
	public long root_id;
	public long left_son;
	public long right_son;
	public long split_feature;
	public double split_feature_value;
	public long estimator_num;
	public long is_root;
	public double node_value;

	public TreeInfo(long root_id, long left_son, long right_son, long split_feature, double split_feature_value,
			long estimator_num, long is_root, double node_value) {
		this.root_id = root_id;
		this.left_son = left_son;
		this.right_son = right_son;
		this.split_feature = split_feature;
		this.split_feature_value = split_feature_value;
		this.estimator_num = estimator_num;
		this.is_root = is_root;
		this.node_value = node_value;
	}

}
