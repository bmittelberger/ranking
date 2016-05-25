package cs276.pa4;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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

/**
 * Implements Pairwise learner that can be used to train SVM
 *
 */
public class PairwiseLearner extends Learner {
	
	String[] TFTYPES = {"url","title","body","header","anchor"};
  private LibSVM model;
  public PairwiseLearner(boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }
    
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
  public PairwiseLearner(double C, double gamma, boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }
    
    model.setCost(C);
    model.setGamma(gamma); // only matter for RBF kernel
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
		
		
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("relevance_score"));
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
				instance.add(rel_data.get(q.query).get(d.url));
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
			for ( int i = 0 ; i < doc_urls.size() - 1 ; i ++){
				Instance inst_i = standardizedDataset.get( doc_to_index.get( doc_urls.get( i )));
				for ( int j = i + 1 ; j < doc_urls.size(); j ++ ){
					double[] inst_j = standardizedDataset.get( doc_to_index.get( doc_urls.get( j ))).toDoubleArray();
					double[] difference = inst_i.toDoubleArray();
					for ( int k = 0 ; k < difference.length ; k ++ ){
						difference[k] = difference[k] - inst_j[k];
					}
					if ( difference[ difference.length - 1 ] < 0 ){
						difference[ difference.length - 1 ] = -1;
					}
					else{
						difference[ difference.length - 1 ] = 1;
					}
					
					Instance difference_inst = new  DenseInstance(1.0, difference);
					finalDSet.add(difference_inst);
				}
			}
		}
		
		finalDSet.setClassIndex(finalDSet.numAttributes() - 1);
		return finalDSet;
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
		System.out.println("num instances: " + dataset.numInstances());
		System.out.println("num attributes: " + dataset.numAttributes());
		model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR,
				LibSVM.TAGS_KERNELTYPE));
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
		
		System.out.println("TESTING NOW");
		System.exit(0);
		/*
		 * @TODO: Your code here
		 */
		return null;
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) {
		/*
		 * @TODO: Your code here
		 */
		return null;
	}

}
