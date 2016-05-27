package cs276.pa4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.stanford.cs276.SmallestWindowScorer;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class PairwiseLearnerExtra extends Learner {

	
	String[] TFTYPES = {"url","title","body","header","anchor"};
	
	double gamma = 0.0078125;
	double C = 0.25;
	boolean linearKernel = false;
	
	  private LibSVM model;
	  public PairwiseLearnerExtra (boolean isLinearKernel){
	    try{
	      model = new LibSVM();
	    } catch (Exception e){
	      e.printStackTrace();
	    }
	    
	    if(isLinearKernel){
	      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
	    }
	  }
	  
	  public PairwiseLearnerExtra (double C, double gamma, boolean isLinearKernel){
	    try{
	      model = new LibSVM();
	    } catch (Exception e){
	      e.printStackTrace();
	    }
	    
	    model.setCost(this.C);
	    model.setGamma(this.gamma); // only matter for RBF kernel
	    if(isLinearKernel){
	      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
	    }
	  }
	  
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
			SmallestWindowScorer swscorer = new SmallestWindowScorer( idfs , queryDict );
			
			List classes = new ArrayList(2); 
			classes.add("pos"); 
			classes.add("neg");  
			
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
			attributes.add(new Attribute("bodylength"));
			attributes.add(new Attribute("BM25"));
			attributes.add(new Attribute("smallestwindow"));
			attributes.add(new Attribute("page_id"));
			attributes.add(new Attribute("url_indicator"));
			
			
			attributes.add(new Attribute("class", classes));
			Instances dataset = null;
			dataset = new Instances("train_dataset", attributes, 0);
			dataset.setClassIndex(dataset.numAttributes() - 1);
			Map<String, Map<String, Integer>> indexes = new HashMap<String, Map<String, Integer>>();
			
			int index = 0;
			for (Query q : train_data.keySet()) {
				List<Document> docs = train_data.get(q);
				if (!indexes.containsKey(q.query)) {
					Map<String, Integer > indexMap = new HashMap<String, Integer>();
					indexes.put(q.query, indexMap);
				}
				for (Document d : docs) {
					ArrayList<Double> instance = get_tfidfs(d, idfs, q);
					instance.add((double)d.page_rank);
					instance.add(get_url_depth( d ));
					instance.add(d.headers != null ? d.headers.size() : 0.0); // count num headers
					instance.add(d.anchors != null ? d.anchors.size() : 0.0); // count num anchors
					instance.add((double)d.body_length); // body length
					instance.add(scorer.getSimScore( d , q ));  //BM25 score from pset 3 
					instance.add(swscorer.getSimScore(d, q));
					instance.add(url_has_page_id( d ));
					instance.add(url_has_word( q, d ));
					instance.add(0.0);
					double[] instancearr = new double[instance.size()];
					for (int i = 0 ; i < instance.size(); i++) {
						instancearr[i] = instance.get(i);
					}
					
					Instance inst = new DenseInstance(1.0, instancearr); 
					dataset.add(inst);
					indexes.get(q.query).put(d.url, index);
					index ++;
				}
			}
			Standardize filter = new Standardize();
			Instances standardizedDataset = null;
			try {
				filter.setInputFormat(dataset);
				standardizedDataset = Filter.useFilter(dataset, filter);
			} catch (Exception e) {
				e.printStackTrace();
			}
			
			Instances finalDSet = new Instances("svm_dataset", attributes, 0);
			
			for (String query : indexes.keySet()) {
				Map<String,Integer> doc_to_index = indexes.get(query);
				List<String> doc_urls = new ArrayList<String>();
				doc_urls.addAll( doc_to_index.keySet() );
				for ( int i = 0 ; i < doc_urls.size()  ; i ++){
					Instance inst_i = standardizedDataset.get( doc_to_index.get( doc_urls.get( i )));
					double rel_i = rel_data.get(query).get(doc_urls.get( i ));
					for ( int j = 0 ; j < doc_urls.size(); j ++ ){
						double[] inst_j = standardizedDataset.get( doc_to_index.get( doc_urls.get( j ))).toDoubleArray();
						double[] difference = inst_i.toDoubleArray();
						for ( int k = 0 ; k < difference.length ; k ++ ){
							difference[k] = difference[k] - inst_j[k];
						}
						double rel_j = rel_data.get(query).get(doc_urls.get( j ));
						if ( rel_i < rel_j ){
							difference[ difference.length - 1 ] = 1; //set to index of "neg"
						}
						else if (rel_i > rel_j ){
							difference[ difference.length - 1 ] = 0; //set to index of "pos"
						} else {
							continue;
						}
						Instance difference_inst = new  DenseInstance(1.0, difference);
						finalDSet.add(difference_inst);
					}
				}
			}
			
			finalDSet.setClassIndex(finalDSet.numAttributes() - 1);
			return finalDSet;
		}
		
		private double url_has_word( Query q, Document d) {
			for (String word : q.queryWords) {
				if (d.url.toLowerCase().contains(word)) {
					return 1.0;
				}
			}
			return 0;
		}
		
		private double get_url_depth( Document d ){
			String[] url_split = d.url.split("/");
			return (double) url_split.length;
		}
		
		private double url_has_page_id( Document d ){
			return d.url.toLowerCase().contains( "page_id" ) ? 1.0 : 0.0 ;
		}


		private ArrayList<Double> get_tfidfs(Document d, Map<String, Double> idfs, Query q) {
			ArrayList<Double> tf_idfs = new ArrayList<Double>();
			for (String type : this.TFTYPES) {
				if (type.equals("url")){
					List<String> urlWords = Parser.parseUrlString(d.url);
					Double urlWeight = this.getListTfIdf(urlWords, idfs, q);
					tf_idfs.add(urlWeight);
				}
				if (type.equals("title")) {
					List<String> titleWords = Parser.parseTitle(d.title);
					Double titleWeight = this.getListTfIdf(titleWords, idfs, q);
					tf_idfs.add(titleWeight);
				}
				if (type.equals("header")) {
					List<String> headerWords = Parser.parseHeaders(d.headers);
					Double headerWeight = this.getListTfIdf(headerWords, idfs, q);
					tf_idfs.add(headerWeight);
				}
				if (type.equals("anchor")) {
					Map<String, Integer> counts = Parser.parseAnchors(d.anchors);
					Double anchorWeight = this.getMapTfIdf(counts, idfs, q);
					tf_idfs.add(anchorWeight);
				}
				if (type.equals("body")) {
					Map<String, Integer> counts = Parser.parseBody(d.body_hits);
					Double bodyWeight = this.getMapTfIdf(counts, idfs, q);
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
			LibSVM model = new LibSVM();
			model.setCost(this.C);
			model.setGamma(this.gamma);
//			model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR,
//					LibSVM.TAGS_KERNELTYPE));
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
			
			Map<Query,List<Document>> train_data = null;
			Map<String, Map<String, Double>> rel_data = null;
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
			SmallestWindowScorer swscorer = new SmallestWindowScorer( idfs , queryDict );
			
			
			
			List classes = new ArrayList(2); 
			classes.add("pos"); 
			classes.add("neg");  
			
			ArrayList<Attribute> attributes = new ArrayList<Attribute>();
			attributes.add(new Attribute("url_w"));
			attributes.add(new Attribute("title_w"));
			attributes.add(new Attribute("header_w"));
			attributes.add(new Attribute("anchor_w"));
			attributes.add(new Attribute("body_w"));
			
			attributes.add(new Attribute("pagerank"));
			attributes.add(new Attribute("urlDepth"));
			attributes.add(new Attribute("numHeaders"));
			attributes.add(new Attribute("numAnchors"));
			attributes.add(new Attribute("bodylength"));
			attributes.add(new Attribute("BM25"));
			attributes.add(new Attribute("smallestwindow"));
			attributes.add(new Attribute("page_id"));
			attributes.add(new Attribute("url_indicator"));
			
			attributes.add(new Attribute("class", classes));
			Instances dataset = null;
			dataset = new Instances("train_dataset", attributes, 0);
			
			TestFeatures t_features = new TestFeatures(dataset);
			
			dataset.setClassIndex(dataset.numAttributes() - 1);
			Map<String, Map<String, Integer>> indexes = new HashMap<String, Map<String, Integer>>();
			
			int index = 0;
			for (Query q : train_data.keySet()) {
				t_features.addQuery(q.query);
				List<Document> docs = train_data.get(q);
				if (!indexes.containsKey(q.query)) {
					Map<String, Integer > indexMap = new HashMap<String, Integer>();
					indexes.put(q.query, indexMap);
				}
				for (Document d : docs) {
					ArrayList<Double> instance = get_tfidfs(d, idfs, q);
					
					instance.add((double)d.page_rank);
					instance.add(get_url_depth( d ));
					instance.add(d.headers != null ? d.headers.size() : 0.0); // count num headers
					instance.add(d.anchors != null ? d.anchors.size() : 0.0); // count num anchors
					instance.add((double)d.body_length); // body length
					instance.add(scorer.getSimScore( d , q ));  //BM25 score from pset 3 
					instance.add(swscorer.getSimScore(d, q));
					instance.add(url_has_page_id( d ));
					instance.add(url_has_word( q, d ));
					instance.add(0.0);
					double[] instancearr = new double[instance.size()];
					for (int i = 0 ; i < instance.size(); i++) {
						instancearr[i] = instance.get(i);
					}
					Instance inst = new DenseInstance(1.0, instancearr); 
					t_features.features.add(inst);
					indexes.get(q.query).put(d.url, index);
					index++;
				}
			}
			Standardize filter = new Standardize();
			Instances standardizedDataset = null;
			try {
				filter.setInputFormat(dataset);
				standardizedDataset = Filter.useFilter(t_features.features, filter);
			} catch (Exception e) {
				e.printStackTrace();
			}
			t_features.features = standardizedDataset;
			
			
			Instances finalDSet = new Instances("svm_dataset", attributes, 0);
			
			index = 0;
			for (String query : indexes.keySet()) {
				Map<String,Integer> doc_to_index = indexes.get(query);
				List<String> doc_urls = new ArrayList<String>();
				doc_urls.addAll( doc_to_index.keySet() );
				for ( int i = 0 ; i < doc_urls.size()  ; i ++){
					Instance inst_i = standardizedDataset.get( doc_to_index.get( doc_urls.get( i )));
					for ( int j = 0 ; j < doc_urls.size(); j ++ ){
						double[] inst_j = standardizedDataset.get( doc_to_index.get( doc_urls.get( j ))).toDoubleArray();
						double[] difference = inst_i.toDoubleArray();
						for ( int k = 0 ; k < difference.length ; k ++ ){
							difference[k] = difference[k] - inst_j[k];
						}
						Instance difference_inst = new  DenseInstance(1.0, difference);
						finalDSet.add(difference_inst);
						String combUrl = doc_urls.get(i) + "|" + doc_urls.get(j);
						t_features.addFeatureIndex(query, combUrl, index);
						index ++;
					}
				}
			}
			
			finalDSet.setClassIndex(finalDSet.numAttributes() - 1);
			t_features.features = finalDSet;
			return t_features;
		}

		@Override
		public Map<String, List<String>> testing(TestFeatures tf,
				Classifier model) {
			
			Instances dataset = tf.features;
			
			
			Map<String, List<String>> rankedQueries = new HashMap<String, List<String>>();
			for (String q : tf.queries) {
				Map<String, Integer> pair_indexes = tf.getIndexes(q);

				
				Set<String> urls = new HashSet<String>();
				Map<String, Integer> comparisons = new HashMap<String, Integer>();
				
				for (String urlPair : pair_indexes.keySet()) {
					Instance toTest = dataset.get(pair_indexes.get(urlPair));
					int classification = -1;
					try {
						classification = (int)model.classifyInstance(toTest);
					} catch (Exception e) {
						e.printStackTrace();
					}
					classification = classification == 0 ? 1 : -1;
					String[] split = urlPair.split("\\|");
					urls.add(split[0]);
					urls.add(split[1]);
					//put both sides of the comparison
					comparisons.put(urlPair, classification);
//					comparisons.put(split[1] + "|" + split[0], 0 - classification);
				}
				final Map<String, Integer> comparisons_2 = new HashMap<String, Integer>(comparisons);
				List<String> urlList = new ArrayList<String>();
				urlList.addAll(urls);
				Collections.sort(urlList, new Comparator<String>() {
					public int compare(String one , String two) {
						  String pair = one + "|" + two;
						  Integer comp = comparisons_2.get(pair);
						  return -comp;
				    }
				});
				
				rankedQueries.put(q, urlList);
			}
			
			
			
			return rankedQueries;
		}

}
