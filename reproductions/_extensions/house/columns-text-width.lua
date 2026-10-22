--[[
  columns-text-width.lua

  On a slide, two columns should span exactly the width that one column does,
  so a pair of blocks below a full-width block lines up with its edges. Beamer's
  columns environment does not do that by default: measured on the house
  canvas, a pair of 48% columns overhung a full-width block by about 6.5pt on
  each side.

  Pandoc's `onlytextwidth` class confines the columns to the text width, and
  the space the widths leave over becomes the gap between them, so 48% and 48%
  leave a 4% gap. This adds the class to every columns div on a slide, so a
  deck never has to remember it. A div that already sets `totalwidth`, or the
  class itself, is left as written.

  Beamer only. Other formats lay columns out by their own rules.
]]

function Div(div)
  if not FORMAT:match 'beamer' then
    return nil
  end
  if div.classes:includes('columns')
      and not div.classes:includes('onlytextwidth')
      and not div.attributes.totalwidth then
    div.classes:insert('onlytextwidth')
    return div
  end
end
