package core.algorithm.lda;


public class LDA implements Runnable{

	@Override
	public void run() {
		LDAOption option = new LDAOption();
		
		option.dir = "F:\\±œ“µ…Ëº∆\\test2";
		option.dfile = "rev_stem.txt";
		option.est = true;  /////
		///option.estc = true;
		option.inf = false;
		option.modelName = "model-final";
		option.K = 15;
		option.niters = 400;
		option.gama[0] = 0.5;
		option.gama[1] = 0.3;
		option.gama[2] = 1;
		option.savestep = 50;
		Estimator estimator = new Estimator();
		estimator.init(option);
		estimator.estimate();
	}

	public static void main(String[] args) {
		new LDA().run();
	}
	
}
