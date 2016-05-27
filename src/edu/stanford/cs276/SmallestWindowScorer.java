package edu.stanford.cs276;

import java.util.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

import cs276.pa4.Document;
import cs276.pa4.Query;
import cs276.pa4.CosineSimilarityScorer;
//import edu.stanford.cs276.util.WordPosition;

/**
 * A skeleton for implementing the Smallest Window scorer in Task 3.
 * Note: The class provided in the skeleton code extends BM25Scorer in Task 2. However, you don't necessarily
 * have to use Task 2. (You could also use Task 1, in which case, you'd probably like to extend CosineSimilarityScorer instead.)
 * Also, feel free to modify or add helpers inside this class.
 */
public class SmallestWindowScorer extends CosineSimilarityScorer {
	
  	double urlweight = 0.2;
  	double titleweight  = 0.4;
  	double bodyweight = 0.1;
  	double headerweight = 0.3;
  	double anchorweight = 0.9;
  	double smoothingBodyLength = 50.0; 
  	double boostFactor = 20.0;
  	double gamma = 0.54;
	
	
	public SmallestWindowScorer(Map<String, Double> idfs, Map<Query,Map<String, Document>> queryDict) {
		super(idfs);
	}

	/**
	 * get smallest window of one document and query pair.
	 * @param d: document
	 * @param q: query
	 */	
	private int getSmallestWindow(Document d, Query q) {
		int smallest = Integer.MAX_VALUE;
		
		
		//URL
//		System.out.println("url");
		List<WordPosition> urlPositions = this.createPositions(d.url.split("[/_\\-.]"), q);
		int checkSmallest = this.smallestWindow(urlPositions, q);
		smallest = Math.min(smallest, checkSmallest);		
		
		//TITLE
//		System.out.println("title");
		List<WordPosition> titlePositions = this.createPositions(d.title.split(" "), q);
		checkSmallest = this.smallestWindow(titlePositions, q);
		smallest = Math.min(smallest, checkSmallest);
		
		//HEADERS
	
		if (d.headers != null) {
//			System.out.println("headers");
			for (String headers : d.headers) {
				List<WordPosition> headerPositions = this.createPositions(headers.split(" "), q);
				checkSmallest = this.smallestWindow(headerPositions, q);
				smallest = Math.min(smallest, checkSmallest);
			}
		}
		
		//BODY POSITIONS
		if (d.body_hits != null) {
//			System.out.println("body");
			List<WordPosition> bodyPositions = this.createBodyPositions(d.body_hits, q);
			checkSmallest = this.smallestWindow(bodyPositions, q);
			smallest = Math.min(smallest, checkSmallest);
		}
		
		//ANCHOR TEXTS
		if (d.anchors != null) {
//			System.out.println("anchor");
			for (String anchorText : d.anchors.keySet()) {
				List<WordPosition> anchorPositions = this.createPositions(anchorText.split(" "), q);
				checkSmallest = this.smallestWindow(anchorPositions, q);
				smallest = Math.min(smallest, checkSmallest);
			}
		}
		
		
		
		return smallest;
	}


	private List<WordPosition> createBodyPositions(Map<String, List<Integer> > bodyPositions, Query q) {
		List<WordPosition> positions = new ArrayList<WordPosition>();
		for (String word : bodyPositions.keySet()) {
			List<Integer> wordPositions = bodyPositions.get(word);
			for (Integer i : wordPositions) {
				WordPosition newPos = new WordPosition(word, i);
				positions.add(newPos);
			}
		}
		Collections.sort(positions);
		return positions;
	}
	
	private String arrToString(String[] fields) {
		String ret = "[";
		for (String i : fields) {
			ret += i + ",";
		}
		ret += "]";
		return ret;
	}
	
	private List<WordPosition> createPositions(String[] fields, Query q) {
//		System.out.println("field: " + arrToString(fields));
		List<WordPosition> positions = new ArrayList<WordPosition>();
		String[] words = fields;
		Set<String> qWords = new HashSet<String>(q.queryWords);
		
		for (int i = 0; i < words.length; i++) {
			if (qWords.contains(words[i])) {
				WordPosition newPos = new WordPosition(words[i],i);
				positions.add(newPos);
			}
		}
		return positions;
	}
	
	private int smallestWindow (List<WordPosition> positions, Query q) {
		int smallest = Integer.MAX_VALUE;
		Set<String> qSet = new HashSet<String>(q.queryWords);
		List<WordPosition> window = new LinkedList<WordPosition>();
		Map<String, Integer> wordCounts = new HashMap<String, Integer>();
		for (String qWord : q.queryWords) {
			wordCounts.put(qWord, 0);
		}
		
		
		for (WordPosition p : positions) {
			if (qSet.contains(p.word)) {
				window.add(p);
				wordCounts.put(p.word, wordCounts.get(p.word) + 1);
			}
			if (window.size() < 1) {
				continue;
			}
			
			//create a set of words that we've collected in the window
			Set<String> found = new HashSet<String>();
			for (String word : wordCounts.keySet()) {
				if (wordCounts.get(word) > 0){
					found.add(word);
				}
			}
			
			if (found.equals(qSet)) {

				//eliminate all the frontmost items to ensure that we get the smallest
				//possible window here
				while (true) {
					if (window.size() < 1) {
						break;
					}
					if (wordCounts.get(window.get(0).word) > 1) {
						wordCounts.put(window.get(0).word, wordCounts.get(window.get(0).word) - 1);
						window.remove(0);
					} else {
						break;
					}
				}
				int end;
				if (window.size() == 0) {
					end = window.get(0).position;
				} else {
					end = window.get(window.size() - 1).position;
				}
				int begin = window.get(0).position;
				int windowLen = end - begin + 1;
				if (windowLen < smallest) {
					smallest = windowLen;
				}
			}
		}
		
		
//		System.out.println("positions: " + positions.toString());
//		System.out.println("final window:" + window.toString());
//		System.out.println("distance: " + smallest + "\n");
		
		return smallest;
	}
	
	/**
	 * get boost score of one document and query pair.
	 * @param d: document
	 * @param q: query
	 */	
	private double getBoostScore (Document d, Query q) {
		int smallestWindow = getSmallestWindow(d, q);
		
		if (smallestWindow == Integer.MAX_VALUE) {
			return 1;
		}
		
		
		double boostScore = 1.0;
		double qSize = (double)(new HashSet<String>(q.queryWords).size());
		double distance = (double)smallestWindow - qSize;
		
		double boostModifier = Math.pow(gamma, distance);
		boostScore = 1 +  ( (boostFactor - 1) * boostModifier) ;
		
		
		return boostScore;
	}
	
	

	
	@Override
	public double getSimScore(Document d, Query q) {
		Map<String,Map<String, Double>> tfs = this.getDocTermFreqs(d,q);
		this.normalizeTFs(tfs, d, q);
		Map<String,Double> tfQuery = getQueryFreqs(q);
		double boost = getBoostScore(d, q);
		double rawScore = this.getNetScore(tfs, q, tfQuery, d);
		return boost * rawScore;
	}

}