package cs276.pa4;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** 
 *  A sample class to store the result
 */
public class TestFeatures {
	/* Test features */
	Instances features;
	
	public TestFeatures(Instances dataset) {
		index_map = new HashMap<String, Map<String, Integer>>();
		queries = new ArrayList<String>();
		features = dataset;
	}
	
	public void addFeatureIndex(String query, String url, int index) {
		if (!index_map.containsKey(query)) {
			Map<String, Integer> mapping = new HashMap<String, Integer>();
			index_map.put(query, mapping);
		}
		index_map.get(query).put(url, index);
	}
	
	public void addQuery(String queryText) {
		queries.add(queryText);
	}
	
	public Map<String, Integer> getIndexes(String query) {
		return index_map.get(query);
	}
	
	/* Associate query-doc pair to its index within FEATURES instances
	 * {query -> {doc -> index}}
	 * 
	 * For example, you can get the feature for a pair of (query, url) using:
	 *   features.get(index_map.get(query).get(url));
	 * */
	Map<String, Map<String, Integer> > index_map;
	List<String> queries;
}
