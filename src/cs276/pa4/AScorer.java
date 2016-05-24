package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
//import java.util.function.BiFunction;

/**
 * 
 * An abstract class for a scorer. 
 * Needs to be extended by each specific implementation of scorers.
 */
public abstract class AScorer {

	// Map: term -> idf
	Map<String,Double> idfs; 

  // Various types of term frequencies that you will need
	String[] TFTYPES = {"url","title","body","header","anchor"};
	
/**
  * Construct an abstract scorer with a map of idfs.
  * @param idfs the map of idf scores
  */
	public AScorer(Map<String,Double> idfs) {
		this.idfs = idfs;
	}
	
/**
	* Score each document for each query.
  * @param d the Document
  * @param q the Query
  */
	public abstract double getSimScore(Document d, Query q);
	
/**
	* Get frequencies for a query.
  * @param q the query to compute frequencies for
  */
	public Map<String,Double> getQueryFreqs(Query q) {

    // queryWord -> term frequency
		Map<String,Double> tfQuery = new HashMap<String, Double>(); 		

		
		for (String word : q.queryWords) {
			if (tfQuery.containsKey(word) ) {
				tfQuery.put(word, tfQuery.get(word) + 1);
			} else {
				tfQuery.put(word, 1.0);
			}
		}
		
		for (String word : tfQuery.keySet()) {
			//Laplace Smoothing tf-idf
			if (!this.idfs.containsKey(word)) {
				tfQuery.put(word, this.idfs.get("$NICKI_MINAJ_NOT$") * tfQuery.get(word));
			} else {
				tfQuery.put(word,this.idfs.get(word) * tfQuery.get(word));
			}
			
		}
		
		return tfQuery;
	}
	
	
	/*
	 * TODO : Your code here
   *        Include any initialization and/or parsing methods
   *        that you may want to perform on the Document fields
   *        prior to accumulating counts.
   *        See the Document class in Document.java to see how
   *        the various fields are represented.
	 */
	
	/**
	 * Accumulate the various kinds of term frequencies 
   * for the fields (url, title, body, header, and anchor).
	 * You can override this if you'd like, but it's likely 
   * that your concrete classes will share this implementation.
   * @param d the Document
   * @param q the Query
	 */
	public Map<String,Map<String, Double>> getDocTermFreqs(Document d, Query q) {

		// Map from tf type -> queryWord -> score
		Map<String,Map<String, Double>> tfs = new HashMap<String,Map<String, Double>>();
		
		/*
		 * TODO : Your code here
     *        Initialize any variables needed
		 */
		
		//{"url","title","body","header","anchor"};
		for (String queryWord : q.queryWords) {
			
			for ( String type : TFTYPES ){
				if (!tfs.containsKey( type )){
					tfs.put( type , new HashMap<String,Double>() );
				}
				if (type.equals( "url" ) ){
					
					String[] urlWords = d.url.split( "[/_\\-.]" ); 
					for (String wrd : urlWords) {
						wrd = wrd.toLowerCase();
					}
					for ( String urlWord : urlWords ){
						if ( queryWord.equals( urlWord ) ){
							Map<String, Double> urlMap = tfs.get("url");
							if ( !urlMap.containsKey( queryWord ) ){
								urlMap.put( queryWord , 1.0 );
							}
							else{
								urlMap.put( queryWord , urlMap.get( queryWord ) + 1 );
							}
						}
					}
				}
				else if (type.equals( "title") ){
					
					String[] titleWords = d.title.split( " " );
					for ( String titleWord : titleWords ){
						if (queryWord.equals( titleWord )){
							Map<String,Double> titleMap = tfs.get("title");
							if ( !titleMap.containsKey(queryWord )){
								titleMap.put( queryWord , 1.0 );
							}
							else{
								titleMap.put( queryWord , titleMap.get( queryWord ) + 1 );
							}
						}
					}
				}
				else if (type.equals( "body") ){
					Map<String,List<Integer>> bodyHits = d.body_hits;
					if (bodyHits == null) {
						continue;
					}
					Map<String,Double> bodyMap = tfs.get("body") ;
					if ( bodyHits.containsKey( queryWord )){
						if ( !bodyMap.containsKey(queryWord ) ) {
							bodyMap.put( queryWord , Double.valueOf( (double) bodyHits.get(queryWord).size()));
						} // else terms are unique so it wont ever be there
					}
				}
				else if (type.equals( "header" ) ){
					List<String> headers = d.headers;
					if (headers == null) {
						continue;
					}
					Map<String,Double> headerMap = tfs.get("header");
					for ( String header : headers ){
						String[] headerWords = header.split( " " );
						for ( String headerWord : headerWords ){
							if ( queryWord.equals( headerWord ) ){
								if (! headerMap.containsKey(queryWord ) ){
									headerMap.put( queryWord , 1.0 );
								}
								else{
									headerMap.put( queryWord , headerMap.get( queryWord ) + 1.0 );
								}
							}
						}
					}
				}
				else { // type.equals( "anchor" ) 
					Map<String, Integer> anchorCounts = d.anchors;
					if (anchorCounts == null ) { 
						continue;
					}
					Map<String, Double> anchorMap = tfs.get( "anchor" );
					for (String anchorText : anchorCounts.keySet()) {
						String[] anchorWords = anchorText.split( " " );
						for (String anchorWord : anchorWords) {
							if (anchorWord.equals(queryWord)) {
								Double anchorCount = new Double(anchorCounts.get(anchorText));
								if (! anchorMap.containsKey(anchorWord)) {
									anchorMap.put(anchorWord, anchorCount);
								} else {
									anchorMap.put(anchorWord, anchorCount + anchorMap.get(anchorWord));
								}
							}
						}
					}
				}
			}
						/*
			 * Your code here
		   * Loop through query terms and accumulate term frequencies. 
       * Note: you should do this for each type of term frequencies,
       * i.e. for each of the different fields.
       * Don't forget to lowercase the query word.
			 */
			
		}
		return tfs;
	}

}
