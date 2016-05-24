package cs276.pa4;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Skeleton code for the implementation of a BM25 Scorer in Task 2.
 */
public class BM25Scorer extends AScorer {

	/*
	*  TODO: You will want to tune these values
	*/
	double urlweight = 0.5;
  	double titleweight  = .9;
  	double bodyweight = 0.3;
  	double headerweight = 0.7;
  	double anchorweight = 0.5;
	
	// BM25-specific weights
	double burl = 0.1;
	double btitle = 0.1;
	double bheader = 0.2;
	double bbody = 0.7;
	double banchor = 0.2;
	
	double k1 = 10;
	double pageRankLambda = 3;
	double pageRankLambdaPrime = .1; // for the log function: add log( pageRankLambdaPrime + page_rank_doc ) to each relevance computation
	
	// query -> url -> document
	Map<Query,Map<String, Document>> queryDict; 

	// BM25 data structures--feel free to modify these
	// Document -> field -> length
	Map<Document,Map<String,Double>> lengths;	

	// field name -> average length
	Map<String,Double> avgLengths;  	

	// Document -> pagerank score
	Map<Document,Double> pagerankScores; 
	
	/**
	   * Construct a BM25Scorer.
	   * @param idfs the map of idf scores
	   * @param queryDict a map of query to url to document
	   */
		public BM25Scorer(Map<String,Double> idfs, Map<Query,Map<String, Document>> queryDict) {
			super(idfs);
			this.queryDict = queryDict;
			this.calcAverageLengths();
		}

	/**
    * Set up average lengths for BM25, also handling PageRank.
    */
	public void calcAverageLengths() {
		lengths = new HashMap<Document,Map<String,Double>>();	// 
		avgLengths = new HashMap<String,Double>();				// map from field Strings to a Double ( average number of terms in a field over entire corpus )
		pagerankScores = new HashMap<Document,Double>();		// the individual page rank scores associated with each document 
		Map<String,ArrayList<Double>> allLengths = new HashMap<String,ArrayList<Double>>();
		
		for ( String tfType : this.TFTYPES ) { 
			allLengths.put( tfType , new ArrayList<Double>() );
		}
		
		for ( Query query : this.queryDict.keySet() ){
			Map<String, Document> documentMap = this.queryDict.get( query );
			for ( String docString : documentMap.keySet() ){
				Document doc = documentMap.get( docString );
				pagerankScores.put( doc , Double.valueOf( (double) doc.page_rank ) );
				for ( String tfType : this.TFTYPES ){
					double fieldLength = getFieldLength( tfType , doc );
					if ( ! lengths.containsKey( doc ) ){
						Map <String,Double> fieldLengths = new HashMap<String,Double>();
						fieldLengths.put( tfType , Double.valueOf( fieldLength ) );
						lengths.put( doc , fieldLengths );
					}
					else{
						Map<String,Double> fieldLengths = lengths.get( doc );
						fieldLengths.put( tfType , Double.valueOf( fieldLength ) );
						lengths.put( doc , fieldLengths );
					}
					allLengths.get( tfType ).add( Double.valueOf( fieldLength ) );			
				}
			}
		}
		
		for ( String tfType : allLengths.keySet() ){
			Double total = 0.0 ;
			for ( Double length : allLengths.get( tfType ) ){
				total += length; 
			}
			Double average = total / (double) allLengths.get( tfType ).size();
			avgLengths.put( tfType , average );
		}
		


	}
	
	private double getWFieldWeight( String type ) {
		if (type.equals("url")) {
			return urlweight;
		} else if (type.equals("title")) {
			return titleweight;
		}else if (type.equals("header")) {
			return headerweight;
		}else if (type.equals("body")) {
			return bodyweight;
		}else { //anchor
			return anchorweight;
		}
		
	}
	
	private double getBFieldWeight( String type ){
		if (type.equals("url")) {
			return burl;
		} else if (type.equals("title")) {
			return btitle;
		}else if (type.equals("header")) {
			return bheader;
		}else if (type.equals("body")) {
			return bbody;
		}else { //anchor
			return banchor;
		}
		
	}
	
	private double getFieldLength( String type,  Document doc ) {

		if (type.equals("url")) {
			String[] urlWords = doc.url.split( "[/_\\-.]" );
			Double urlLength = Double.valueOf( (double) urlWords.length );
			return urlLength; 
		} else if (type.equals("title")) {
			String[] titleWords = doc.title.split( " " );
			Double titleLength = Double.valueOf( (double) titleWords.length );
			return titleLength;
		}else if (type.equals("header")) {
			List<String> headers = doc.headers;
			Double headersLength = 0.0;
			if (headers != null) {
				for ( String header : headers ){
					headersLength += (double) header.split( " " ).length;
				}
			}
			return headersLength;
		}else if (type.equals("body")) {
			Map<String,List<Integer>> bodyHits = doc.body_hits;
			Double bodyLength = 0.0; // if bodyHits null then return the default 0.0 value
			if ( bodyHits != null ){
				for ( String term : bodyHits.keySet() ){
					bodyLength += (double) bodyHits.get( term ).size(); 
				}
			}
			return bodyLength;
		}else { //anchor
			Map<String, Integer> anchors = doc.anchors;
			Double anchorsLength = 0.0;
			if ( anchors != null) {
				for ( String anchor : anchors.keySet() ){
					double anchorCount = anchors.get( anchor ).doubleValue() ;
					double textLength = (double) anchor.split( " " ).length;
					anchorsLength += Double.valueOf( textLength * anchorCount );
				}
			}
			return anchorsLength;
		}
	}

  /**
    * Get the net score. 
    * @param tfs the term frequencies
    * @param q the Query 
    * @param tfQuery
    * @param d the Document
    * @return the net score
    */
	public double getNetScore(Map<String,Map<String, Double>> tfs, Query q, Map<String,Double> tfQuery,Document d) {

		double score = 0.0;
		
		/*
		 * TODO : Your code here
		 *        Use equation 5 in the writeup to compute the overall score
		 *        of a document d for a query q.
		 */
		Map<String,Double> wdt = new HashMap<String,Double>();
		
		// compute equation four values , the weight sums of BM25 field-relative word frequencies
		for ( String word : q.queryWords ){
			
			for ( String tfType : this.TFTYPES ){
				Double weight = getWFieldWeight( tfType );
				
				if ( tfs.get( tfType ).containsKey( word ) ) {
					if ( ! wdt.containsKey( word  ) ){
						wdt.put( word , weight*tfs.get( tfType ).get( word ) );
					}
					else{
						Double currentWeight = wdt.get( word );
						wdt.put( word , weight*tfs.get( tfType ).get( word ) + currentWeight );
					}
				}	
			}
		}
		
		for ( String word : wdt.keySet() ){
			Double idf = 0.0;
			if ( this.idfs.containsKey( word ) ){
				idf = this.idfs.get( word );
			}
			else{
				idf = this.idfs.get( "$NICKI_MINAJ_NOT$" );
			}
			Double numerator = wdt.get( word );
			Double denominator = k1 + wdt.get( word );
			score += ( numerator / denominator ) * idf ;
		}
		
		// add the page rank feature function value
		// log function works better than division function
		
		//log function
		score += pageRankLambda * Math.log( pageRankLambdaPrime  + d.page_rank );
		
		//division function
		//score += ( (double) d.page_rank ) / ( pageRankLambdaPrime + (double) d.page_rank ); 
		
		
		return score;
	}

  /**
    * Do BM25 Normalization.
    * @param tfs the term frequencies
    * @param d the Document
    * @param q the Query
    */
	public void normalizeTFs(Map<String,Map<String, Double>> tfs , Document d , Query q) {
		/*
		 * TODO : Your code here
		 *        Use equation 3 in the writeup to normalize the raw term frequencies
		 *        in fields in document d.
		 */
		// matt note : tfs is already relative to the specific document d 
		
		for ( String tfType : this.TFTYPES ){
			Map<String,Double> termCounts = tfs.get( tfType );
			
			Double averageFieldLength = avgLengths.get( tfType );
			Double docFieldLength = lengths.get( d ).get( tfType );
			double fieldWeight = getBFieldWeight( tfType );
			Double lengthDiv = docFieldLength / averageFieldLength;
			Double denominator = ( 1.0 + fieldWeight*( lengthDiv - 1 ) );
			
			for ( String term : termCounts.keySet() ){
				termCounts.put( term , termCounts.get( term ) / denominator );
			}
		}
		
	}
	
	
	
	
	/**
	    * Write the tuned parameters of BM25 to file.
	    * Only used for grading purpose, you should NOT modify this method.
	    * @param filePath the output file path.
	    */
	private void writeParaValues(String filePath) {
		try {
			File file = new File(filePath);
			if (!file.exists()) {
				file.createNewFile();
			}
			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			String[] names = {"urlweight", "titleweight", "bodyweight", 
        "headerweight", "anchorweight", "burl", "btitle", 
        "bheader", "bbody", "banchor", "k1", "pageRankLambda", "pageRankLambdaPrime"};
			double[] values = {this.urlweight, this.titleweight, this.bodyweight, 
        this.headerweight, this.anchorweight, this.burl, this.btitle, 
        this.bheader, this.bbody, this.banchor, this.k1, this.pageRankLambda, 
        this.pageRankLambdaPrime};
			BufferedWriter bw = new BufferedWriter(fw);
			for (int idx = 0; idx < names.length; ++ idx) {
				bw.write(names[idx] + " " + values[idx]);
				bw.newLine();
			}
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
  /**
    * Get the similarity score.
    * @param d the Document
    * @param q the Query
    * @return the similarity score
    */
	public double getSimScore(Document d, Query q) {
		Map<String,Map<String, Double>> tfs = this.getDocTermFreqs(d,q);
		this.normalizeTFs(tfs, d, q);
		Map<String,Double> tfQuery = getQueryFreqs(q);

		// Write out the tuned BM25 parameters
    // This is only used for grading purposes.
		// You should NOT modify the writeParaValues method.
		writeParaValues("bm25Para.txt");
		return getNetScore(tfs,q,tfQuery,d);
	}
	
}
