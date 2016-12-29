package util;

import java.util.HashMap;

public class ParamReader {
	public static int readInt(String key, HashMap<String, Object> map) throws ParamException {
		if (!map.containsKey(key))
			throw new ParamException("Params doesn't have key " + key);
		return Integer.valueOf(map.get(key).toString());
	}

	public static double readDouble(String key, HashMap<String, Object> map) throws ParamException {
		if (!map.containsKey(key))
			throw new ParamException("Params doesn't have key " + key);
		return Double.valueOf(map.get(key).toString());
	}

	public static String readString(String key, HashMap<String, Object> map) throws ParamException {
		if (!map.containsKey(key))
			throw new ParamException("Params doesn't have key " + key);
		return map.get(key).toString();
	}
}
