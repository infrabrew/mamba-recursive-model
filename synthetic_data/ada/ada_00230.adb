package Matrix_Operations is
   type Matrix is array (Positive range <>, Positive range <>) of Float;

   function Multiply(A, B : Matrix) return Matrix;
   function Transpose(M : Matrix) return Matrix;
   procedure Print(M : Matrix);
end Matrix_Operations;

package body Matrix_Operations is
   function Multiply(A, B : Matrix) return Matrix is
      Result : Matrix(A'Range(1), B'Range(2));
   begin
      for I in A'Range(1) loop
         for J in B'Range(2) loop
            Result(I, J) := 0.0;
            for K in A'Range(2) loop
               Result(I, J) := Result(I, J) + A(I, K) * B(K, J);
            end loop;
         end loop;
      end loop;
      return Result;
   end Multiply;

   function Transpose(M : Matrix) return Matrix is
      Result : Matrix(M'Range(2), M'Range(1));
   begin
      for I in M'Range(1) loop
         for J in M'Range(2) loop
            Result(J, I) := M(I, J);
         end loop;
      end loop;
      return Result;
   end Transpose;

   procedure Print(M : Matrix) is
   begin
      for I in M'Range(1) loop
         for J in M'Range(2) loop
            Put(Float'Image(M(I, J)) & " ");
         end loop;
         New_Line;
      end loop;
   end Print;
end Matrix_Operations;