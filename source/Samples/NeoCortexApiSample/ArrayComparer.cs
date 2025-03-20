using System;
using System.Collections.Generic;
using System.Linq;

public class ArrayComparer : IEqualityComparer<int[]>
{
    public bool Equals(int[] x, int[] y)
    {
        if (x == null || y == null)
            return false;
        return x.SequenceEqual(y);
    }

    public int GetHashCode(int[] obj)
    {
        if (obj == null)
            throw new ArgumentNullException(nameof(obj));

        unchecked
        {
            int hash = 17;
            foreach (var val in obj)
            {
                hash = hash * 31 + val.GetHashCode();
            }
            return hash;
        }
    }
}
