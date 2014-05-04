package core.algorithm.lda;

import java.io.Serializable;


public class Inferencer implements Serializable{	

	private static final long serialVersionUID = 466563090503055129L;
	
	// Train model
	public Model trnModel;
	public Dictionary globalDict;
	private LDAOption option;
	
	private Model newModel;
	public int niters = 100;
	
	//-----------------------------------------------------
	// Init method
	//-----------------------------------------------------
	public boolean init(LDAOption option){
		this.option = option;
		trnModel = new Model();
		
		if (!trnModel.initEstimatedModel(option))
			return false;		
		
		globalDict = trnModel.data.localDict;
		computeTrnTheta();
		computeTrnPhi();
		computeTrnPi();
		
		return true;
	}
	
	//inference new model ~ getting data from a specified dataset
	public Model inference(LDADataset newData){
		//////System.out.println("init new model");
		Model newModel = new Model();
		
		newModel.initNewModel(option, newData, trnModel);		
		this.newModel = newModel;		
		
		/////System.out.println("Sampling " + niters + " iteration for inference!");		
		for (newModel.liter = 1; newModel.liter <= niters; newModel.liter++){
			//System.out.println("Iteration " + newModel.liter + " ...");
			
			// for all newz_i
			for (int m = 0; m < newModel.M; ++m){
				for (int n = 0; n < newModel.data.docs[m].length; n++){
					// (newz_i = newz[m][n]
					// sample from p(z_i|z_-1,w)
					int res[] = infSampling(m, n);
					newModel.z[m].set(n, res[0]);
					newModel.l[m].set(n, res[1]);
					
				}
			}//end foreach new doc
			
		}// end iterations
		
		//////System.out.println("Gibbs sampling for inference completed!");
		
		computeNewTheta();
		computeNewPhi();
		computeNewPi();
		newModel.liter--;
		return this.newModel;
	}
	
	public Model inference(String [] strs, int I){
		//System.out.println("inference");
		//Model newModel = new Model();   //////////////// never read????
		
		//System.out.println("read dataset");
		LDADataset dataset = LDADataset.readDataSet(strs, globalDict,I);
		
		return inference(dataset);
	}
	
	//inference new model ~ getting dataset from file specified in option
	public Model inference(){	
		//System.out.println("inference");
		
		newModel = new Model();
		if (!newModel.initNewModel(option, trnModel)) return null;
		
		/////System.out.println("Sampling " + niters + " iteration for inference!");
		
		for (newModel.liter = 1; newModel.liter <= niters; newModel.liter++){
			//System.out.println("Iteration " + newModel.liter + " ...");
			
			// for all newz_i
			for (int m = 0; m < newModel.M; ++m){
				for (int n = 0; n < newModel.data.docs[m].length; n++){
					// (newz_i = newz[m][n]
					// sample from p(z_i|z_-1,w)
					int []res = infSampling(m, n);
					newModel.z[m].set(n, res[0]);
					newModel.l[m].set(n, res[1]);
				}
			}//end foreach new doc
			
		}// end iterations
		
		/////System.out.println("Gibbs sampling for inference completed!");		
		/////System.out.println("Saving the inference outputs!");
		
		computeNewTheta();
		computeNewPhi();
		newModel.liter--;
		newModel.saveModel(newModel.dfile + "." + newModel.modelName);		
		
		return newModel;
	}
	
	/**
	 * do sampling for inference
	 * m: document number
	 * n: word number?
	 */
	protected int[] infSampling(int m, int n){
		// remove z_i from the count variables
		int topic = trnModel.z[m].get(n);
		int sentiment = trnModel.l[m].get(n);
		int it = trnModel.data.docs[m].item;
		int _w = newModel.data.docs[m].words[n];
		int w = newModel.data.lid2gid.get(_w);
		
		trnModel.nw[topic][sentiment][w] -= 1;
		trnModel.nr[m][topic][sentiment] -= 1;
		trnModel.ni[it][topic] -= 1;
		trnModel.nwsum[topic][sentiment] -= 1;
		trnModel.nrsum[m][topic] -= 1;
		trnModel.nisum[it] -= 1;
		
		double Vbeta = trnModel.V * trnModel.beta;
		double Kalpha = trnModel.K * trnModel.alpha;
		double Sgama = trnModel.gama[0]+ trnModel.gama[1];
		
		//do multinominal sampling via cumulative method
		for (int k = 0; k < trnModel.K; k++)
			for (int s = 0; s < trnModel.S; s++)
			{
				trnModel.p[k][s] = (trnModel.nw[k][s][w] + trnModel.beta)/(trnModel.nwsum[k][s] + Vbeta) *
						(trnModel.nr[m][k][s] + trnModel.gama[s])/(trnModel.nrsum[m][k] + Sgama)*
						(trnModel.ni[it][k]+trnModel.alpha)/(trnModel.nisum[it] + Kalpha);
			}
		
		// cumulate multinomial parameters
		for (int k = 0; k < trnModel.K; k++)
		{
			if(k > 0) trnModel.p[k][0] += trnModel.p[k-1][trnModel.S - 1];
			for (int s = 1; s < trnModel.S; s++)
				trnModel.p[k][s] += trnModel.p[k][s-1];
		}
		// scaled sample because of unnormalized p[]
		double u = Math.random() * trnModel.p[trnModel.K - 1][trnModel.S - 1];
		
		for (topic = 0; topic < trnModel.K; topic++)
			for (sentiment = 0; sentiment < trnModel.S; sentiment++)
			{
				if (trnModel.p[topic][sentiment] > u) //sample topic w.r.t distribution p
					break;
			}	
		
		// add newly estimated z_i to count variables
		trnModel.nw[topic][sentiment][w] += 1;
		trnModel.nr[m][topic][sentiment] += 1;
		trnModel.ni[it][topic] += 1;
		trnModel.nwsum[topic][sentiment] += 1;
		trnModel.nrsum[m][topic] += 1;
		trnModel.nisum[it] += 1;
		
		int []res = new int[2];
		res[0] = topic;
		res[1] = sentiment;
 		return res;
	}
	
	protected void computeNewTheta(){
		for (int i = 0; i < newModel.I; i++){
			for (int k = 0; k < newModel.K; k++){
				newModel.theta[i][k] = (newModel.ni[i][k] + newModel.alpha) / 
						(newModel.nisum[i] + newModel.K * newModel.alpha);
			}//end foreach topic
		}//end foreach new document
	}
	
	protected void computeNewPhi(){
		for (int k = 0; k < newModel.K; k++){
			for(int s = 0; s < newModel.S; s++)
				for (int _w = 0; _w < newModel.V; _w++){
				Integer id = newModel.data.lid2gid.get(_w);
				
				if (id != null){
					newModel.phi[k][s][_w] = (trnModel.nw[k][s][id] + newModel.nw[k][s][_w] + newModel.beta) / (newModel.nwsum[k][s] + newModel.nwsum[k][s] + trnModel.V * newModel.beta);
				}
			}//end foreach word
		}// end foreach topic
	}
	
	public void computeNewPi(){
		for (int m = 0; m < newModel.M; m++)
			for(int k = 0; k < newModel.K; k++)
				for(int s = 0; s < newModel.S; s++)
				newModel.pi[m][k][s] = (newModel.nr[m][k][s] + newModel.gama[s])/(newModel.nrsum[m][k] + newModel.gama[0] + newModel.gama[1]);
	}
	
	protected void computeTrnTheta(){
		for (int i = 0; i < trnModel.I; i++){
			for (int k = 0; k < trnModel.K; k++){
				trnModel.theta[i][k] = (trnModel.ni[i][k] + trnModel.alpha) / 
						(trnModel.nisum[i] + trnModel.K * trnModel.alpha);
			}
		}
	}
	
	protected void computeTrnPhi(){
		for (int k = 0; k < trnModel.K; k++)
			for (int s = 0; s < trnModel.S; s++){
				for (int w = 0; w < trnModel.V; w++){
				trnModel.phi[k][s][w] = (trnModel.nw[k][s][w] + trnModel.beta) / (trnModel.nwsum[k][s] + trnModel.V * trnModel.beta);
			}
		}
	}
	
	public void computeTrnPi(){
		for (int m = 0; m < trnModel.M; m++)
			for(int k = 0; k < trnModel.K; k++)
				for(int s = 0; s < trnModel.S; s++)
				trnModel.pi[m][k][s] = (trnModel.nr[m][k][s] + trnModel.gama[s])/(trnModel.nrsum[m][k] + trnModel.gama[0] + trnModel.gama[1]);
	}
}


