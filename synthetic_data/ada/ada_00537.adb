with Ada.Containers.Vectors;

package Data_Structures is
   type Element is record
      ID : Integer;
      Value : Float;
      Name : String(1..50);
   end record;

   package Element_Vectors is new Ada.Containers.Vectors
     (Index_Type   => Natural,
      Element_Type => Element);

   procedure Add_Element(V : in out Element_Vectors.Vector; E : Element);
   function Find_By_ID(V : Element_Vectors.Vector; ID : Integer) return Element;
end Data_Structures;

package body Data_Structures is
   procedure Add_Element(V : in out Element_Vectors.Vector; E : Element) is
   begin
      V.Append(E);
   end Add_Element;

   function Find_By_ID(V : Element_Vectors.Vector; ID : Integer) return Element is
   begin
      for E of V loop
         if E.ID = ID then
            return E;
         end if;
      end loop;
      raise Constraint_Error with "Element not found";
   end Find_By_ID;
end Data_Structures;