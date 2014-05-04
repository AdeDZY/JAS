package core.algorithm.lda;

import java.io.File;

public class Estimator {
	
	// output model
	protected Model trnModel;
	LDAOption option;
	
	public boolean init(LDAOption option){
		this.option = option;
		trnModel = new Model();
		
		if (option.est){
			if (!trnModel.initNewModel(option))
				return false;
			trnModel.data.localDict.writeWordMap(option.dir + File.separator + option.wordMapFileName);
		}
		else if (option.estc){
			if (!trnModel.initEstimatedModel(option))
				return false;
		}
		
		return true;
	}
	
	public void estimate(){
		
		System.out.println("Sampling " + trnModel.niters + " iteration!");
		int lastIter = trnModel.liter;
		for (trnModel.liter = lastIter + 1; trnModel.liter < trnModel.niters + lastIter; trnModel.liter++){
			System.out.println("Iteration " + trnModel.liter + " ...");
			// for all z_i
			for (int m = 0; m < trnModel.M; m++){				
				for (int n = 0; n < trnModel.data.docs[m].length; n++){
					// z_i = z[m][n]
					// sample from p(z_i|z_-i, w)
					int []res = sampling(m, n);
					trnModel.z[m].set(n, res[0]);
					trnModel.l[m].set(n, res[1]);
				}// end for each word
			}// end for each document
			
			if (option.savestep > 0){
				if (trnModel.liter % option.savestep == 0){
					System.out.println("Saving the model at iteration " + trnModel.liter + " ...");
					computeTheta();
					computePhi();
					computePi();
					trnModel.saveModel("model-" + LDAUtils.zeroPad(trnModel.liter, 5));
				}
			}
		}// end iterations		
		
		System.out.println("Gibbs sampling completed!\n");
		System.out.println("Saving the final model!\n");
		
		computeTheta();
		computePhi();
		trnModel.liter--;
		trnModel.saveModel("model-final");
	}
	
	/**
	 * Do sampling
	 * @param m document number
	 * @param n word number
	 * @return topic id
	 */
	public int[] sampling(int m, int n){
		// remove z_i from the count variable
		int topic = trnModel.z[m].get(n);
		int sentiment = trnModel.l[m].get(n);
		int w = trnModel.data.docs[m].words[n];
		int it = trnModel.data.docs[m].item;
		
		trnModel.nw[topic][sentiment][w] -= 1;
		trnModel.nr[m][topic][sentiment] -= 1;
		trnModel.ni[it][topic] -= 1;
		trnModel.nwsum[topic][sentiment] -= 1;
		trnModel.nrsum[m][topic] -= 1;
		trnModel.nisum[it] -= 1;
		
		double Vbeta[] = new double[3];
		for(int t = 0; t < 3; t++)
		{
			Vbeta[t] = 0.0;
			for(int j = 0; j < 3; j++)
				Vbeta[t] += trnModel.data.localDict.n[j] * trnModel.beta[t][j];
		}
		double Kalpha = trnModel.K * trnModel.alpha;
		double Sgama = 0.0;
		for(int s = 0; s < trnModel.S; s++)
			Sgama += trnModel.gama[s];
		
		//do multinominal sampling via cumulative method
		for (int k = 0; k < trnModel.K; k++)
			for (int s = 0; s < trnModel.S; s++)
			{
				trnModel.p[k][s] = (trnModel.nw[k][s][w] + trnModel.beta[s][trnModel.data.localDict.getSent(w)])/(trnModel.nwsum[k][s] + Vbeta[s]) *
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
		{
			for (sentiment = 0; sentiment < trnModel.S; sentiment++)
			{
				if (trnModel.p[topic][sentiment] > u) //sample topic w.r.t distribution p
					break;
			}	
			if(sentiment < trnModel.S)
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
	
	public void computeTheta(){
		for (int i = 0; i < trnModel.I; i++){
			for (int k = 0; k < trnModel.K; k++){
				trnModel.theta[i][k] = (trnModel.ni[i][k] + trnModel.alpha) / 
						(trnModel.nisum[i] + trnModel.K * trnModel.alpha);
			}
		}
	}
	
	public void computePhi(){
		double Vbeta[] = new double[3];
		for(int t = 0; t < 3; t++)
		{
			Vbeta[t] = 0.0;
			for(int j = 0; j < 3; j++)
				Vbeta[t] += trnModel.data.localDict.n[j] * trnModel.beta[t][j];
		}
		
		for (int k = 0; k < trnModel.K; k++)
			for (int s = 0; s < trnModel.S; s++){
				for (int w = 0; w < trnModel.V; w++){
				trnModel.phi[k][s][w] = (trnModel.nw[k][s][w] + trnModel.beta[s][trnModel.data.localDict.getSent(w)]) / (trnModel.nwsum[k][s] + Vbeta[s]);
				}
			}
		}
	public void computePi(){
		for (int m = 0; m < trnModel.M; m++)
			for(int k = 0; k < trnModel.K; k++)
				for(int s = 0; s < trnModel.S; s++)
				trnModel.pi[m][k][s] = (trnModel.nr[m][k][s] + trnModel.gama[s])/(trnModel.nrsum[m][k] + trnModel.gama[0] + trnModel.gama[1]);
	}
	

}
