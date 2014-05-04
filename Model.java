package core.algorithm.lda;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.StringTokenizer;
import java.util.Vector;

public class Model implements Serializable{	
	
	//---------------------------------------------------------------
	//	Class Variables
	//---------------------------------------------------------------
	private static final long serialVersionUID = 3094739584562527534L;
	
	public static String tassignSuffix;	 //suffix for topic assignment file
	public static String thetaSuffix;		//suffix for theta (topic - item distribution) file
	public static String phiSuffix;		//suffix for phi file (topic -sentiment - word distribution) file
	public static String piSuffix;		//suffix for pi file (review-topic-sentiment distribution)
	public static String othersSuffix; 	//suffix for containing other parameters
	public static String twordsSuffix;		//suffix for file containing words-per-topics
	
	//---------------------------------------------------------------
	//	Model Parameters and Variables
	//---------------------------------------------------------------
	
	public String wordMapFile; 		//file that contain word to id map
	public String trainlogFile; 	//training log file	
	
	public String dir;
	public String dfile;
	public String modelName;
	public int modelStatus; 		//see Constants class for status of model
	public LDADataset data;			// link to a dataset
	
	public int M; //number of documents (total number of reviews)
	public int V; //vocabulary size
	public int K; //number of topics
	public int S; //number of sentiment
	public int I; //number of Items (businesses)
	public double alpha;
	public double [][]beta;
	public double []gama; //LDA  hyperparameters
	public int niters; //number of Gibbs sampling iteration
	public int liter; //the iteration at which the model was saved	
	public int savestep; //saving period
	public int twords; //print out top words per each topic
	public int withrawdata;
	
	// Estimated/Inferenced parameters
	public double [][] theta; //theta: item - topic distributions, size I x K
	public double [][][] phi; // phi: topic-sentiment-word distributions, size K x S x V
	public double [][][] pi; //pi: doc-topic-sentiment distributions, size M x  K x S
	
	// Temp variables while sampling
	public Vector<Integer> [] z; //topic assignments for words
	public Vector<Integer> [] l; //sentiment assignments for words
	protected int [][][] nw; //nw[k][s][i]: number of instances of word/term i assigned to topic k, sentiment s, size V x K x S
	protected int [][][] nr; //nr[m][k][s]: number of words in doc m, topic k assigned to sentiment s, size M x Rm x K x S
	protected int [][] ni; //ni[i][k]: number of words in item i, topic k, size M x K
	protected int [][] nwsum; //nwsum[k][s]: total number of words assigned to topic k, sentiment s, size K x S
	protected int [][] nrsum; //ndsum[m][k]: total number of words assigned to doc m, topic k, size M x Rm x K
	protected int [] nisum; //nisum[i]: total number of words in item i, size M
	
	// temp variables for sampling
	protected double [][] p; 
	
	//---------------------------------------------------------------
	//	Constructors
	//---------------------------------------------------------------	

	public Model(){
		setDefaultValues();	
	}
	
	/**
	 * Set default values for variables
	 */
	public void setDefaultValues(){
		
		wordMapFile = "wordmap.txt";
		trainlogFile = "trainlog.txt";
		tassignSuffix = ".tassign";
		thetaSuffix = ".theta";
		phiSuffix = ".phi";
		othersSuffix = ".others";
		twordsSuffix = ".twords";
		piSuffix = ".pi";	
		
		dir = "./";
		dfile = "trndocs.dat";
		modelName = "model-final";
		modelStatus = LDAOption.MODEL_STATUS_UNKNOWN;		
		
		M = 0;
		V = 0;
		K = 100;
		S = 10;
		alpha = 50.0 / K;
		beta = new double[S][S];
		gama = new double[S];
		gama[0] = 1;
		gama[1] = 0.1;
		gama[3] = 3;
		niters = 2000;
		liter = 0;
		
		z = null;
		nw = null;
		nr = null;
		ni = null;
		nwsum = null;
		nrsum = null;
		nisum = null;
		theta = null;
		phi = null;
		pi = null;
		
	}
	
	//---------------------------------------------------------------
	//	I/O Methods
	//---------------------------------------------------------------
	/**
	 * read other file to get parameters
	 */
	protected boolean readOthersFile(String otherFile){
		//open file <model>.others to read:
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader(otherFile));
			String line;
			while((line = reader.readLine()) != null){
				StringTokenizer tknr = new StringTokenizer(line,"= \t\r\n");
				
				int count = tknr.countTokens();
				if(count != 2)
					continue;
				
				String optstr = tknr.nextToken();
				String optval = tknr.nextToken();
				
				if (optstr.equalsIgnoreCase("alpha")){
					alpha = Double.parseDouble(optval);					
				}
				else if (optstr.equalsIgnoreCase("beta")){
					//beta = Double.parseDouble(optval);
				}
				else if (optstr.equalsIgnoreCase("gama1")){
					gama[0]= Double.parseDouble(optval);
				}
				else if (optstr.equalsIgnoreCase("gama2")){
					gama[1] = Double.parseDouble(optval);
				}
				else if (optstr.equalsIgnoreCase("ntopics")){
					K = Integer.parseInt(optval);
				}
				else if (optstr.equalsIgnoreCase("nsentiments")){
					S = Integer.parseInt(optval);
				}
				else if (optstr.equalsIgnoreCase("nitems")){
					I = Integer.parseInt(optval);
				}
				else if (optstr.equalsIgnoreCase("liter")){
					liter = Integer.parseInt(optval);
				}
				else if (optstr.equalsIgnoreCase("nwords")){
					V = Integer.parseInt(optval);
				}
				else if (optstr.equalsIgnoreCase("ndocs")){
					M = Integer.parseInt(optval);
				}
			}
			
			reader.close();
		}
		catch (Exception e){
			System.out.println("Error while reading other file:" + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	@SuppressWarnings("unchecked")
	protected boolean readTAssignFile(String tassignFile){
		try {
			int i,j;
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(tassignFile), "GB18030"));
			
			String line;
			z = new Vector[M];			
			data = new LDADataset(M, I);
			data.V = V;			
			for (i = 0; i < M; i++){
				line = reader.readLine();
				StringTokenizer tknr = new StringTokenizer(line, " \t\r\n");
				
				int length = tknr.countTokens();
				
				Vector<Integer> words = new Vector<Integer>();
				Vector<Integer> topics = new Vector<Integer>();
				Vector<Integer> sentiments = new Vector<Integer>();
				
				int item = Integer.parseInt(tknr.nextToken());
				for (j = 0; j < length; j++){
					String token = tknr.nextToken();
					
					StringTokenizer tknr2 = new StringTokenizer(token, ":");
					if (tknr2.countTokens() != 3){
						System.out.println("Invalid word-topic assignment line\n");
						return false;
					}
					
					words.add(Integer.parseInt(tknr2.nextToken()));
					topics.add(Integer.parseInt(tknr2.nextToken()));
					sentiments.add(Integer.parseInt(tknr2.nextToken()));
				}//end for each topic assignment
				
				//allocate and add new document to the corpus
				Document doc = new Document(words, item);
				data.setDoc(doc, i);
				
				//assign values for z
				z[i] = new Vector<Integer>();
				l[i] = new Vector<Integer>();
				for (j = 0; j < topics.size(); j++){
					z[i].add(topics.get(j));
					l[i].add(sentiments.get(j));
				}
				
			}//end for each doc
			
			reader.close();
		}
		catch (Exception e){
			System.out.println("Error while loading model: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * load saved model
	 */
	public boolean loadModel(){
		if (!readOthersFile(dir + File.separator + modelName + othersSuffix))
			return false;
		
		if (!readTAssignFile(dir + File.separator + modelName + tassignSuffix))
			return false;
		
		// read dictionary
		Dictionary dict = new Dictionary();
		if (!dict.readWordMap(dir + File.separator + wordMapFile))
			return false;
			
		data.localDict = dict;
		
		return true;
	}
	
	/**
	 * Save word-topic assignments for this model
	 */
	public boolean saveModelTAssign(String filename){
		int i, j;
		
		try{
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			
			//write docs with topic assignments for words
			for (i = 0; i < data.M; i++){
				for (j = 0; j < data.docs[i].length; ++j){
					writer.write(data.docs[i].words[j] + ":" + z[i].get(j) + ":" + l[i].get(j)+" ");					
				}
				writer.write("\n");
			}
				
			writer.close();
		}
		catch (Exception e){
			System.out.println("Error while saving model tassign: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save theta (topic distribution) for this model
	 */
	public boolean saveModelTheta(String filename){
		try{
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			for (int i = 0; i < I; i++){
				for (int j = 0; j < K; j++){
					//writer.write(theta[i][j] + " ");
					writer.write(baoliu(theta[i][j], 4) + " ");
				}
				writer.write("\n");
			}
			writer.close();
		}
		catch (Exception e){
			System.out.println("Error while saving topic distribution file for this model: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}


	/**
	 * Save word-topic distribution
	 */
	
	public boolean saveModelPhi(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			
			for (int k = 0; k < K; k++)
				for(int s = 0; s < S; s++){
					for (int w = 0; w < V; w++){
						writer.write(baoliu(phi[k][s][w], 4) + " ");
				}
				writer.write("\n");
			}
			writer.close();
		}
		catch (Exception e){
			System.out.println("Error while saving word-topic distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	public boolean saveModelPi(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			
			for (int m = 0; m < M; m++)
				for(int k = 0; k < K; k++){
					for (int s = 0; s < S; s++){
						writer.write(baoliu(pi[m][k][s], 4) + " ");
				}
				writer.write("\n");
			}
			writer.close();
		}
		catch (Exception e){
			System.out.println("Error while saving word-topic distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	/**
	 * Save other information of this model
	 */
	public boolean saveModelOthers(String filename){
		try{
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			
			writer.write("alpha=" + alpha + "\n");
			//writer.write("beta=" + beta + "\n");
			writer.write("ntopics=" + K + "\n");
			writer.write("ndocs=" + M + "\n");
			writer.write("nwords=" + V + "\n");
			writer.write("nsentiments=" + S + "\n");
			writer.write("nitems=" + I + "\n");
			writer.write("liters=" + liter + "\n");
			
			writer.close();
		}
		catch(Exception e){
			System.out.println("Error while saving model others:" + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save model the most likely words for each topic
	 */
	public boolean saveModelTwords(String filename){
		try{
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
					new FileOutputStream(filename), "utf-8"));
			
			if (twords > V){
				twords = V;
			}
			
			for (int k = 0; k < K; k++)
			{
				writer.write("Topic " + k + "th:\n");
				for (int s = 0; s < S; s++)
				{
					List<Pair> wordsProbsList = new ArrayList<Pair>(); 
					
					for (int w = 0; w < V; w++)
					{
						Pair p = new Pair(w, phi[k][s][w], false);
						wordsProbsList.add(p);
					}//end foreach word
				
					//print topic-sentiment				
					writer.write("Sentiment " + s + "th:\n");
					Collections.sort(wordsProbsList);
					for (int i = 0; i < twords; i++)
					{
						if (data.localDict.contains((Integer)wordsProbsList.get(i).first))
						{
							String word = data.localDict.getWord((Integer)wordsProbsList.get(i).first);
							writer.write("\t" + word + " " + wordsProbsList.get(i).second + "\n");
						}
					}
				}//end foreach sentiment 
			}//end foreach topic			
						
			writer.close();
		}
		catch(Exception e){
			System.out.println("Error while saving model twords: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save model
	 */
	public boolean saveModel(String modelName){
		if (!saveModelTAssign(dir + File.separator + modelName + tassignSuffix)){
			return false;
		}
		
		if (!saveModelOthers(dir + File.separator + modelName + othersSuffix)){			
			return false;
		}
		
		if (!saveModelTheta(dir + File.separator + modelName + thetaSuffix)){
			return false;
		}
		
		if (!saveModelPhi(dir + File.separator + modelName + phiSuffix)){
			return false;
		}
		
		if (!saveModelPhi(dir + File.separator + modelName + piSuffix)){
			return false;
		}
		
		
		if (twords > 0){
			if (!saveModelTwords(dir + File.separator + modelName + twordsSuffix))
				return false;
		}
		return true;
	}
	
	//---------------------------------------------------------------
	//	Init Methods
	//---------------------------------------------------------------
	/**
	 * initialize the model
	 */
	protected boolean init(LDAOption option){		
		if (option == null)
			return false;
		
		modelName = option.modelName;
		K = option.K;
		S = option.S;
		
		alpha = option.alpha;
		if (alpha < 0.0)
			alpha = 50.0 / K;
		
		for(int t = 0; t < S; t++)
			for (int j = 0; j < S; j++)
			beta[t][j] = option.beta[t][j];
		
		for(int t = 0; t < S; t++)
			gama[t] = option.gama[t];
		
		niters = option.niters;
		
		dir = option.dir;
		if (dir.endsWith(File.separator))
			dir = dir.substring(0, dir.length() - 1);
		
		dfile = option.dfile;
		twords = option.twords;
		wordMapFile = option.wordMapFileName;
		
		return true;
	}
	
	/**
	 * Init parameters for estimation
	 */
	@SuppressWarnings("unchecked")
	public boolean initNewModel(LDAOption option){
		if (!init(option))
			return false;
		
		int m, n, w, k, s, i;		
		p = new double[K][S];		
		
		data = LDADataset.readDataSet(dir + File.separator + dfile);
		if (data == null){
			System.out.println("Fail to read training data!\n");
			return false;
		}
		
		//+ allocate memory and assign values for variables		
		M = data.M;
		V = data.V;
		I = data.I;
		dir = option.dir;
		savestep = option.savestep;
		
		// K: from command line or default value
	    // alpha, beta: from command line or default values
	    // niters, savestep: from command line or default values

		nw = new int[K][S][V];
		for (k = 0; k < K; k++)
			for(s = 0; s < S; s++)
				for (w = 0; w < V; w++)
				nw[k][s][w] = 0;

		nr = new int[M][K][S];
		for (m = 0; m < M; m++)
			for (k = 0; k < K; k++)
				for(s = 0; s < S; s++)
					nr[m][k][s] = 0;
		
		ni = new int[I][K];
		for (i = 0; i < I; i++)
			for (k = 0; k < K; k++)
					ni[i][k] = 0;

		
		nwsum = new int[K][S];
		for (k = 0; k < K; k++)
			for(s = 0; s < S; s++)
				nwsum[k][s] = 0;
		
		nrsum = new int[M][K];
		for (m = 0; m < M; m++)
			for(k = 0; k < K; k++)
				nrsum[m][k] = 0;
		
		nisum = new int[I];
		for (i = 0; i < I; i++)
			nisum[i] = 0;
		
		z = new Vector[M];
		l = new Vector[M];
		for (m = 0; m < data.M; m++){
			int N = data.docs[m].length;
			z[m] = new Vector<Integer>();
			l[m] = new Vector<Integer>();
			//initilize for z and l
			for (n = 0; n < N; n++){
				int topic = (int)Math.floor(Math.random() * K);
				z[m].add(topic);
				//int sentiment = (int)Math.floor(Math.random() * S);
				int sentiment = 2;
				Integer wordID = data.docs[m].words[n]; 
				if(data.localDict.getSent(wordID) == 0)
						sentiment = 0;
				if(data.localDict.getSent(wordID) == 1)
					sentiment = 1;
				else
				{
					double r = Math.random();
					if (r < 0.1)
						sentiment = 0;
					else if (r >= 0.9)
						sentiment = 1;
					else
						sentiment = 2;
				}
				l[m].add(sentiment);

				// number of instances of word assigned to topic j
				nw[topic][sentiment][data.docs[m].words[n]] += 1;
				// number of words in document i assigned to topic j
				nr[m][topic][sentiment] += 1;
				//
				ni[data.docs[m].item][topic] += 1;
				// total number of words assigned to topic j
				nwsum[topic][sentiment] += 1;
				//
				nrsum[m][topic] += 1;
				
			}
			
			// total number of words in item i
			nisum[data.docs[m].item] += N;
		}
		
		theta = new double[I][K];		
		phi = new double[K][S][V];
		pi = new double[M][K][S];
		
		return true;
	}
	
	/**
	 * Init parameters for inference
	 * @param newData DataSet for which we do inference
	 */
	@SuppressWarnings("unchecked")
	public boolean initNewModel(LDAOption option, LDADataset newData, Model trnModel){
		if (!init(option))
			return false;
		
		int m, n, w, k, s, i;		
		p = new double[K][S];		
		
		K = trnModel.K;
		S = trnModel.S;
		alpha = trnModel.alpha;
		beta = trnModel.beta;		
		for(int t = 0; t < S; t++)
			gama[t] = option.gama[t];
		////System.out.println("K:" + K);
		
		data = newData;
		
		//+ allocate memory and assign values for variables		
		M = data.M;
		V = data.V;
		I = data.I;
		dir = option.dir;
		savestep = option.savestep;
		////System.out.println("M:" + M);
		////System.out.println("V:" + V);
		
		// K: from command line or default value
	    // alpha, beta: from command line or default values
	    // niters, savestep: from command line or default values

		nw = new int[K][S][V];
		for (k = 0; k < K; k++)
			for(s = 0; s < S; s++)
				for (w = 0; w < V; w++)
				nw[k][s][w] = 0;

		nr = new int[M][K][S];
		for (m = 0; m < M; m++)
			for (k = 0; k < K; k++)
				for(s = 0; s < S; s++)
					nr[m][k][s] = 0;
		
		ni = new int[I][K];
		for (i = 0; i < I; i++)
			for (k = 0; k < K; k++)
					ni[i][k] = 0;

		
		nwsum = new int[K][S];
		for (k = 0; k < K; k++)
			for(s = 0; s < S; s++)
				nwsum[k][s] = 0;
		
		nrsum = new int[M][K];
		for (m = 0; m < M; m++)
			for(k = 0; k < K; k++)
				nwsum[m][K] = 0;
		
		nisum = new int[I];
		for (i = 0; i < I; i++)
			nisum[i] = 0;
		
		z = new Vector[M];
		l = new Vector[M];
		for (m = 0; m < data.M; m++){
			int N = data.docs[m].length;
			z[m] = new Vector<Integer>();
			
			//initilize for z and l
			for (n = 0; n < N; n++){
				int topic = (int)Math.floor(Math.random() * K);
				z[m].add(topic);
				//int sentiment = (int)Math.floor(Math.random() * S);
				int sentiment = 2;
				Integer wordID = data.docs[m].words[n]; 
				if(data.localDict.getSent(wordID) == 0)
						sentiment = 0;
				if(data.localDict.getSent(wordID) == 1)
					sentiment = 1;
				else
				{
					double r = Math.random();
					if (r < 0.1)
						sentiment = 0;
					else if (r >= 0.9)
						sentiment = 1;
					else
						sentiment = 2;
				}
				l[m].add(sentiment);

				// number of instances of word assigned to topic j
				nw[topic][sentiment][data.docs[m].words[n]] += 1;
				// number of words in document i assigned to topic j
				nr[m][topic][sentiment] += 1;
				//
				ni[data.docs[m].item][topic] += 1;
				// total number of words assigned to topic j
				nwsum[topic][sentiment] += 1;
				//
				nrsum[m][topic] += 1;
				
			}
			
			// total number of words in item i
			nisum[data.docs[m].item] += N;
		}
		
		theta = new double[I][K];		
		phi = new double[K][S][V];
		pi = new double[M][K][S];
		
		
		return true;
	}
	
	/**
	 * Init parameters for inference
	 * reading new dataset from file
	 */
	public boolean initNewModel(LDAOption option, Model trnModel){
		if (!init(option))
			return false;
		
		LDADataset dataset = LDADataset.readDataSet(dir + File.separator + dfile, trnModel.data.localDict);
		if (dataset == null){
			System.out.println("Fail to read dataset!\n");
			return false;
		}
		
		return initNewModel(option, dataset , trnModel);
	}
	
	/**
	 * init parameter for continue estimating or for later inference
	 */
	public boolean initEstimatedModel(LDAOption option){
		if (!init(option))
			return false;
		
		int m, n, w, k, s, i;
		
		p = new double[K][S];
		
		// load model, i.e., read z and trndata
		if (!loadModel()){
			System.out.println("Fail to load word-topic assignment file of the model!\n");
			return false;
		}
		
		System.out.println("Model loaded: " + "alpha: " + alpha + " | beta: " + beta + " | M: " + M + " | V: " + V);
		System.out.println("\talpha:" + alpha);
		System.out.println("\tbeta:" + beta);
		System.out.println("\tM:" + M);
		System.out.println("\tV:" + V);		
		
		nw = new int[K][S][V];
		for (k = 0; k < K; k++)
			for(s = 0; s < S; s++)
				for (w = 0; w < V; w++)
				nw[k][s][w] = 0;

		nr = new int[M][K][S];
		for (m = 0; m < M; m++)
			for (k = 0; k < K; k++)
				for(s = 0; s < S; s++)
					nr[m][k][s] = 0;
		
		ni = new int[I][K];
		for (i = 0; i < I; i++)
			for (k = 0; k < K; k++)
					ni[i][k] = 0;

		
		nwsum = new int[K][S];
		for (k = 0; k < K; k++)
			for(s = 0; s < S; s++)
				nwsum[k][s] = 0;
		
		nrsum = new int[M][K];
		for (m = 0; m < M; m++)
			for(k = 0; k < K; k++)
				nwsum[m][K] = 0;
		
		nisum = new int[I];
		for (i = 0; i < I; i++)
			nisum[i] = 0;
	    
	    for (m = 0; m < data.M; m++){
	    	int N = data.docs[m].length;
	    	
	    	// assign values for nw, nd, nwsum, and ndsum
	    	for (n = 0; n < N; n++){
	    		w = data.docs[m].words[n];
	    		int topic = (Integer)z[m].get(n);
	    		int sentiment = (Integer)l[m].get(n);
	    		
	    		// number of instances of word assigned to topic j
				nw[topic][sentiment][data.docs[m].words[n]] += 1;
				// number of words in document i assigned to topic j
				nr[m][topic][sentiment] += 1;
				//
				ni[data.docs[m].item][topic] += 1;
				// total number of words assigned to topic j
				nwsum[topic][sentiment] += 1;
				//
				nrsum[m][topic] += 1;
						
	    	}
	    	// total number of words in item i
	    	nisum[data.docs[m].item] += N;
	    }
	    
		theta = new double[I][K];		
		phi = new double[K][S][V];
		pi = new double[M][K][S];
	    dir = option.dir;
		savestep = option.savestep;
	    
		return true;
	}
	

	/**
	 * ��������������dС����nλ
	 * 
	 * @param dout
	 * @param n
	 * @return
	 */
	public static double baoliu(double d, int n) {
		double p = Math.pow(10, n);
		return Math.round(d * p) / p;
	}
	
}



