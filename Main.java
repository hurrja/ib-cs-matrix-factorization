import java.util.concurrent.ThreadLocalRandom;

public class Main
{
  public static void main (String[] args)
  {
    // hyperparameters
    final int F = 2; // number of features
    final double GAMMA = 0.0001; // sgd step size
    final double LAMBDA = 0.002; // regularization weight for user weights
    final double MU = LAMBDA; // regularization weight for degrees
    final int NUM_SGD_ROUNDS = 10000; // termination condition

    final int[][] R = {{3, 2, 3, -1, -1},
      {-1, 2, 2, 2, -1},
      {3, 4, -1, 4, 2},
      {2, 2, 4, -1, 1},
      {2, -1, 3, 2, 1},
      {-1, -1, 3, 1, 1}};
    final int NUM_USERS = R.length;
    final int NUM_ITEMS = R [0].length;

    double[][] W = new double [NUM_USERS][F];
    initializeOnes (W);
    double[][] D = new double [F][NUM_ITEMS];
    initializeOnes (D);

    {
      // local block to hide i and j
      int i = 0, j = 0;
      for (int r = 0; r < NUM_SGD_ROUNDS; r++)
      {
        if (r % (NUM_SGD_ROUNDS / 10) == 0) // log 10 times
        {
          // log progress
          double g = 0; // objective function
          for (int k = 0; k < NUM_USERS; k++)
            for (int l = 0; l < NUM_ITEMS; l++)
            {
              if (R [k][l] >= 0)
              {
                double d = R [i][j] - predicted (i, j, W, D);
                g += d * d;
              }
            }
          
          g += LAMBDA * l2Norm (W);
          g += MU * l2Norm (D);
          
          System.out.println ("objective: " + String.format ("%.3g", g));
        }
        
        boolean selected = false;
        while (!selected)
        {
          i = randIndex (NUM_USERS);
          j = randIndex (NUM_ITEMS);
          if (R [i][j] >= 0)
            selected = true;
        }

        double p = predicted (i, j, W, D); // predicted value
        double e = R [i][j] - p; // prediction error
      
        for (int k = 0; k < F; k++)
        {
          double dw = 2 * (-D [k][j] * e + LAMBDA * W [i][k]); // dg/dw
          W [i][k] -= GAMMA * dw;
        
          double dd = 2 * (-D [k][j] * e + LAMBDA * W [i][k]); // dg/dd
          D [k][j] -= GAMMA * dd;
        }
      }
    }
    
    double[][] P = new double [NUM_USERS][NUM_ITEMS];
    
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
                        
  private static int randIndex (int size)
  {
    return ThreadLocalRandom.current().nextInt(size);
  }
}
