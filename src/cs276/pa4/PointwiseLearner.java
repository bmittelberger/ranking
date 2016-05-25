package cs276.pa4;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Implements point-wise learner that can be used to implement logistic regression
 *
 */
public class PointwiseLearner extends Learner {

	
	String[] TFTYPES = {"url","title","body","header","anchor"};
	@Override
	public Instances extractTrainFeatures(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {
		
		/*
		 * @TODO: Below is a piece of sample code to show 
		 * you the basic approach to construct a Instances 
		 * object, replace with your implementation. 
		 */
		
		Map<Query,List<Document>> train_data = null;
		Map<String, Map<String, Double>> rel_data = null;
		try {
			train_data = Util.loadTrainData(train_data_file);
			rel_data = Util.loadRelData(train_rel_file);
		} catch (Exception e) {
			e.printStackTrace();
		}
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("relevance_score"));
		Instances dataset = null;
		dataset = new Instances("train_dataset", attributes, 0);
		
		for (Query q : train_data.keySet()) {
			List<Document> docs = train_data.get(q);
			for (Document d : docs) {
				double instance[] = get_tfidfs(d, idfs, q);
				instance[5] = rel_data.get(q.query).get(d.url);
				Instance inst = new DenseInstance(1.0, instance); 
				dataset.add(inst);
			}
			
		}
		
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);
		
		return dataset;
	}

	
	//"url","title","body","header","anchor"
	private double[] get_tfidfs(Document d, Map<String, Double> idfs, Query q) {
//		Map<String, Double> tf_idfs = new HashMap<String, Double>();
		double[] tf_idfs = new double[6];
		for (String type : this.TFTYPES) {
			if (type.equals("url")){
				List<String> urlWords = Parser.parseUrlString(d.url);
				Double urlWeight = this.getListTfIdf(urlWords, idfs, q);
				tf_idfs[0] = urlWeight;
			}
			if (type.equals("title")) {
				List<String> titleWords = Parser.parseTitle(d.title);
				Double titleWeight = this.getListTfIdf(titleWords, idfs, q);
				tf_idfs[1] = titleWeight;
			}
			if (type.equals("header")) {
				List<String> headerWords = Parser.parseHeaders(d.headers);
				Double headerWeight = this.getListTfIdf(headerWords, idfs, q);
				tf_idfs[2] = headerWeight;
			}
			if (type.equals("anchor")) {
				Map<String, Integer> counts = Parser.parseAnchors(d.anchors);
				Double anchorWeight = this.getMapTfIdf(counts, idfs, q);
				tf_idfs[3] = anchorWeight;
			}
			if (type.equals("body")) {
				Map<String, Integer> counts = Parser.parseBody(d.body_hits);
				Double bodyWeight = this.getMapTfIdf(counts, idfs, q);
				tf_idfs[4] = bodyWeight;
			}
		}
		return tf_idfs;
	}
	
	private Double getListTfIdf(List<String> words, Map<String, Double> idfs, Query q) {
		Map<String, Double> counts = new HashMap<String, Double> ();
		for (String word : words) {
			if (counts.containsKey(word)) {
				counts.put(word, counts.get(word) + 1.0);
			} else {
				counts.put(word, 1.0);
			}
		}
		Double weight = 0.0;
		for (String qWord : q.queryWords) {
			if (counts.containsKey(qWord) && idfs.containsKey(qWord)) {
				weight += counts.get(qWord) * idfs.get(qWord);
			}
		}
		return weight;
	}
	
	private Double getMapTfIdf(Map<String, Integer> counts, Map<String, Double> idfs, Query q) {
		Double weight = 0.0;
		for (String qWord : q.queryWords) {
			if (counts.containsKey(qWord) && idfs.containsKey(qWord)) {
				weight += counts.get(qWord) * idfs.get(qWord);
			}
		}
		return weight;
	}
	
	
	@Override
	public Classifier training(Instances dataset) {
		LinearRegression model = new LinearRegression();
		try {
			model.buildClassifier(dataset);
		} catch(Exception e) {
			e.printStackTrace();
		}
		return model;
	}

	@Override
	public TestFeatures extractTestFeatures(String test_data_file,
			Map<String, Double> idfs) {
		
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("relevance_score"));
		Instances dataset = null;
		dataset = new Instances("train_dataset", attributes, 0);
		
		TestFeatures t_features = new TestFeatures(dataset);
		Map<Query,List<Document>> train_data = null;
		try {
			train_data = Util.loadTrainData(test_data_file);			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		int index = 0;
		for (Query q : train_data.keySet()) {
			t_features.addQuery(q.query);
			List<Document> docs = train_data.get(q);
			for (Document d : docs) {
				double instance[] = get_tfidfs(d, idfs, q);
				instance[5] = 1.0;
//				System.out.print("Url: " + d.url + " - [");
//				for (int i = 0; i < instance.length; i++) {
//					System.out.print(instance[i] + ", ");
//				}
//				System.out.println("]");
				Instance inst = new DenseInstance(1.0, instance); 
				t_features.features.add(inst);
				t_features.addFeatureIndex(q.query, d.url, index);
				index ++;
			}	
		}
		
		return t_features;
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) {
		
		Map<String, List<String>> rankedQueries = new HashMap<String, List<String>>();
		for (String q : tf.queries) {
			Map<String, Integer> q_indexes = tf.getIndexes(q);
			List<Pair<String, Double>> ranks = new ArrayList<Pair<String, Double> > ();
			for (String url : q_indexes.keySet()) {
				try {
					Double rel = model.classifyInstance(tf.features.instance(q_indexes.get(url)));
					Pair<String, Double> relPair = new Pair<String, Double>(url, rel);
					ranks.add(relPair);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			Collections.sort(ranks, new Comparator<Pair<String,Double> >() {
				public int compare(Pair<String, Double> one , Pair<String,Double> two) {
			          double diff =  one.getSecond()*1000 - two.getSecond()*1000;
			          return (int)-diff;
			    }
			});
			List<String> finalRanks = new ArrayList<String>();
			for (Pair<String, Double> p : ranks) {
				finalRanks.add(p.getFirst());
			}
			rankedQueries.put(q, finalRanks);
		}
		
		return rankedQueries;
	}

}
