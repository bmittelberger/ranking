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

//import edu.stanford.cs276.AScorer;
//import edu.stanford.cs276.BM25Scorer;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Implements pointwise learner with extra features.
 *
 */
public class PointwiseLearnerExtra extends Learner {

	
	String[] TFTYPES = {"url","title","header","anchor","body"};
	@Override
	public Instances extractTrainFeatures(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {
		
		
		Map<Query,List<Document>> train_data = null;
		Map<String, Map<String, Double>> rel_data = null;
		try {
			train_data = Util.loadTrainData(train_data_file);
			rel_data = Util.loadRelData(train_rel_file);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
		Map<Query, Map<String, Document>> queryDict = new HashMap<Query, Map<String,Document>>();
		for (Query q_ : train_data.keySet()) {
			queryDict.put( q_ , new HashMap<String,Document>() );
			List<Document> docs_ = train_data.get(q_ );
			for (Document d : docs_) {
				queryDict.get(q_).put( d.url , d );
			}
		}
		
		AScorer scorer = new BM25Scorer(idfs, queryDict);
		
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("body_w"));
		
		//Made changes to original here.
		attributes.add(new Attribute("pagerank"));
		attributes.add(new Attribute("urlDepth"));
		attributes.add(new Attribute("numHeaders"));
		attributes.add(new Attribute("numAnchors"));
		attributes.add(new Attribute("bodyLength"));
		attributes.add(new Attribute("totalLength"));
		
		
		attributes.add(new Attribute("relevance_score"));
	
		
		Instances dataset = null;
		dataset = new Instances("train_dataset", attributes, 0);
		
		// finally, what are small window features ?
		for (Query q : train_data.keySet()) {
			List<Document> docs = train_data.get(q);
			for (Document d : docs) {
				ArrayList<Double> instance = get_tfidfs(d, idfs, q);
				instance.add((double)d.page_rank);
				instance.add(get_url_depth( d ));
				instance.add(d.headers != null ? d.headers.size() : 0.0); // count num headers
				instance.add(d.anchors != null ? d.anchors.size() : 0.0); // count num anchors
				instance.add((double)d.body_length); // body length
				instance.add(scorer.getSimScore( d , q ));  //BM25 score from pset 3 
				instance.add(rel_data.get(q.query).get(d.url));
				double[] instancearr = new double[instance.size()];
				for (int i = 0 ; i < instance.size(); i++) {
					instancearr[i] = instance.get(i);
				}
				Instance inst = new DenseInstance(1.0, instancearr); 
				dataset.add(inst);
			}
			
		}
		
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);
		
		return dataset;
	}
	
	private double get_url_depth( Document d ){
		String[] url_split = d.url.split("/");
		return (double) url_split.length;
	}

	
	//"url","title","body","header","anchor"
	private ArrayList<Double> get_tfidfs(Document d, Map<String, Double> idfs, Query q) {
//		Map<String, Double> tf_idfs = new HashMap<String, Double>();
		//double[] tf_idfs = new double[12]; // adding features 
		ArrayList<Double> tf_idfs = new ArrayList<Double>();
		for (String type : this.TFTYPES) {
			if (type.equals("url")){
				List<String> urlWords = Parser.parseUrlString(d.url);
				Double urlWeight = this.getListTfIdf(urlWords, idfs, q);
//				tf_idfs[0] = urlWeight;
				tf_idfs.add(urlWeight);
			}
			if (type.equals("title")) {
				List<String> titleWords = Parser.parseTitle(d.title);
				Double titleWeight = this.getListTfIdf(titleWords, idfs, q);
//				tf_idfs[1] = titleWeight;
				tf_idfs.add(titleWeight);
			}
			if (type.equals("header")) {
				List<String> headerWords = Parser.parseHeaders(d.headers);
				Double headerWeight = this.getListTfIdf(headerWords, idfs, q);
//				tf_idfs[2] = headerWeight;
				tf_idfs.add(headerWeight);
			}
			if (type.equals("anchor")) {
				Map<String, Integer> counts = Parser.parseAnchors(d.anchors);
				Double anchorWeight = this.getMapTfIdf(counts, idfs, q);
//				tf_idfs[3] = anchorWeight;
				tf_idfs.add(anchorWeight);
			}
			if (type.equals("body")) {
				Map<String, Integer> counts = Parser.parseBody(d.body_hits);
				Double bodyWeight = this.getMapTfIdf(counts, idfs, q);
//				tf_idfs[4] = bodyWeight;
				tf_idfs.add(bodyWeight);
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
		//Made changes to original here.
		attributes.add(new Attribute("pagerank"));
		attributes.add(new Attribute("urlDepth"));
		attributes.add(new Attribute("numHeaders"));
		attributes.add(new Attribute("numAnchors"));
		attributes.add(new Attribute("bodyLength"));
		attributes.add(new Attribute("totalLength"));
		
		
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
		
		Map<Query, Map<String, Document>> queryDict = new HashMap<Query, Map<String,Document>>();
		for (Query q_ : train_data.keySet()) {
			queryDict.put( q_ , new HashMap<String,Document>() );
			List<Document> docs_ = train_data.get(q_ );
			for (Document d : docs_) {
				queryDict.get(q_).put( d.url , d );
			}
		}
		
		AScorer scorer = new BM25Scorer(idfs, queryDict);
		
		
		
		int index = 0;
		for (Query q : train_data.keySet()) {
			t_features.addQuery(q.query);
			List<Document> docs = train_data.get(q);
			for (Document d : docs) {
				ArrayList<Double> instance = get_tfidfs(d, idfs, q);
				instance.add((double)d.page_rank);
				instance.add(get_url_depth( d ));
				instance.add(d.headers != null ? d.headers.size() : 0.0); // count num headers
				instance.add(d.anchors != null ? d.anchors.size() : 0.0); // count num anchors
				instance.add((double)d.body_length); // body length
				instance.add(scorer.getSimScore( d , q ));  //BM25 score from pset 3 
				instance.add(1.0);
				double[] instancearr = new double[instance.size()];
				for (int i = 0 ; i < instance.size(); i++) {
					instancearr[i] = instance.get(i);
				}
				Instance inst = new DenseInstance(1.0, instancearr); 
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
