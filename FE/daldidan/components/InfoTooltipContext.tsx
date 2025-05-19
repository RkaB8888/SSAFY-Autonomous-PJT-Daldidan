// InfoTooltipContext.tsx
import React, { createContext, useContext, useState } from 'react';

interface InfoTooltipContextType {
  hasShown: boolean;
  setHasShown: (shown: boolean) => void;
}

const InfoTooltipContext = createContext<InfoTooltipContextType>({
  hasShown: false,
  setHasShown: () => {},
});

export const InfoTooltipProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [hasShown, setHasShown] = useState(false);

  return (
    <InfoTooltipContext.Provider value={{ hasShown, setHasShown }}>
      {children}
    </InfoTooltipContext.Provider>
  );
};

export const useInfoTooltip = () => useContext(InfoTooltipContext);
