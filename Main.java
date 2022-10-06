public class Main
{
  public static void main (String[] args)
  {
    // hyperparameters
    final int F = 2; // number of features
    final double GAMMA = 0.001; // sgd step size
    final double LAMBDA = 0.01; // regularization weight for user weights
    final double MU = LAMBDA; // regularization weight for degrees

    // termination condition: this many rounds over entire data
    final int NUM_SGD_ROUNDS = 1000; 
    
    // observed values
    final int[][] R = {{3, 2, 3, -1, -1},
                       {-1, 2, 2, 2, -1},
                       {3, 4, -1, 4, 2},
                       {2, 2, 4, -1, 1},
                       {2, -1, 3, 2, 1},
                       {-1, -1, 3, 1, 1}};
    final int NUM_USERS = R.length;
    final int NUM_ITEMS = R [0].length;

    // factor matrices
    double[][] W = new double [NUM_USERS][F]; // users to features
    initializeOnes (W);
    double[][] D = new double [F][NUM_ITEMS]; // features to items
    initializeOnes (D);

    // loop NUM_SGD_ROUNDS over entire data
    for (int r = 0; r < NUM_SGD_ROUNDS; r++)
    {
      if (r % (NUM_SGD_ROUNDS / 10) == 0) // log progress 10 times
      {
        double g = 0; // value of objective function
        for (int i = 0; i < NUM_USERS; i++)
          for (int j = 0; j < NUM_ITEMS; j++)
          {
            if (R [i][j] >= 0)
            {
              double d = R [i][j] - predicted (i, j, W, D); // prediction error
              g += d * d;
            }
          }
          
        // L2 norms of matrices
        g += LAMBDA * l2Norm (W);
        g += MU * l2Norm (D);
          
        System.out.println ("objective: " + String.format ("%.3g", g));
      }
        
      // modify factor matrices for all nonnegative observed values
      for (int i = 0; i < NUM_USERS; i++)
        for (int j = 0; j < NUM_ITEMS; j++)
          if (R [i][j] >= 0)
          {
            double e = R [i][j] - predicted (i, j, W, D); // prediction error
            
            for (int k = 0; k < F; k++)
            {
              double dw = 2 * (-D [k][j] * e + LAMBDA * W [i][k]); // dg/dw
              W [i][k] -= GAMMA * dw; // gradient descent
              
              double dd = 2 * (-W [i][k] * e + MU * D [k][j]); // dg/dd
              D [k][j] -= GAMMA * dd;
            }
          }
    }
    
    // output predicted values
    for (int i = 0; i < NUM_USERS; i++)
    {
      for (int j = 0; j < NUM_ITEMS; j++)
        System.out.print (String.format ("%.2f", predicted (i, j, W, D)) + " ");
      System.out.println ("");
    }
  }

  private static double l2Norm (double[][] A)
  {
    double n = 0; // norm
    for (double[] r : A)
      for (double v : r)
        n += v * v;

    return n;
  }

  private static void initializeOnes (double[][] A)
  {
    for (int i = 0; i < A.length; i++)
      for (int j = 0; j < A [0].length; j++)
        A [i][j] = 1;
  }
  
  private static double predicted (int i, int j, double[][] W, double[][] D)
  {
    assert (W [0].length == D.length); // inner dimensions match

    double p = 0; // predicted value
    for (int k = 0; k < D.length; k++)
      p += W [i][k] * D [k][j];

    return p;
  }
}
