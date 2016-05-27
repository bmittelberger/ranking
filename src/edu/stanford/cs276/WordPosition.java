package edu.stanford.cs276;

//import cspa4.WordPosition;

public class WordPosition implements Comparable<WordPosition>{
	
	public String word;
	public Integer position;
	
	public WordPosition(String word, Integer position) {
		this.word = word;
		this.position = position;
	}
	

	@Override
	public int compareTo(WordPosition o) {
		if (this.position < o.position) {
			return -1;
		}
		return 1;
	}
	
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("<" + this.word + "," + this.position + ">");
		return sb.toString();
	}
}